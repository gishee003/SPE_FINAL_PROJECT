import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


DEFAULT_TARGET_COLUMN = "Exited"
DEFAULT_DROP_COLUMNS = {"Surname", "id", "CustomerId", "RowNumber"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_serializable_number(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _extract_positive_class_shap_values(shap_values) -> np.ndarray:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    arr = np.array(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def _build_explainer(model, background_df: pd.DataFrame):
    if shap is None:
        raise ImportError("shap is not installed. Install dependencies from requirements.txt.")
    try:
        return shap.Explainer(model, background_df)
    except Exception:
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        background = shap.sample(background_df, min(100, len(background_df)))
        return shap.Explainer(predict_fn, background)


def _install_sklearn_pickle_compat_shims():
    """
    Backward/forward-compat shim for sklearn pipeline pickles.
    Some model artifacts reference private sklearn classes that may not
    exist in another runtime (for example `_RemainderColsList`).
    """
    try:
        from sklearn.compose import _column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        # If shim injection fails, pickle.load will still raise a useful error.
        pass


class DriftRCAExplainer:
    def __init__(
        self,
        model_path: str,
        train_csv_path: str,
        test_csv_path: str,
        reference_path: Optional[str] = None,
        target_column: str = DEFAULT_TARGET_COLUMN,
    ):
        self.model_path = model_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.reference_path = reference_path
        self.target_column = target_column

        self.model = None
        self.train_df = None
        self.test_df = None
        self.feature_columns: List[str] = []
        self.baseline_importance: Dict[str, float] = {}

    def load(self):
        _install_sklearn_pickle_compat_shims()
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as exc:
            msg = str(exc)
            if "_RemainderColsList" in msg or "sklearn.compose._column_transformer" in msg:
                raise RuntimeError(
                    "Model deserialization failed due to scikit-learn version mismatch. "
                    "Run RCA in the same runtime as training, or retrain the model with the current environment."
                ) from exc
            raise RuntimeError(f"Failed to load model from {self.model_path}: {exc}") from exc

        if not os.path.exists(self.train_csv_path):
            raise FileNotFoundError(f"Training CSV not found at {self.train_csv_path}")
        if not os.path.exists(self.test_csv_path):
            raise FileNotFoundError(f"Test CSV not found at {self.test_csv_path}")

        self.train_df = pd.read_csv(self.train_csv_path)
        self.test_df = pd.read_csv(self.test_csv_path)
        self.feature_columns = [
            col
            for col in self.train_df.columns
            if col != self.target_column and col not in DEFAULT_DROP_COLUMNS
        ]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_columns].copy()

    def build_baseline(self):
        if self.model is None:
            self.load()
        train_features = self._prepare_features(self.train_df)
        
        # Optimize: Use a representative sample for baseline importance to save time/CPU
        sample_size = min(200, len(train_features))
        baseline_sample = train_features.sample(n=sample_size, random_state=42) if len(train_features) > sample_size else train_features

        explainer = _build_explainer(self.model, train_features)
        shap_values = explainer(baseline_sample)
        positive_class_shap = _extract_positive_class_shap_values(shap_values)
        mean_abs = np.mean(np.abs(positive_class_shap), axis=0)
        self.baseline_importance = {
            feature: float(mean_abs[idx]) for idx, feature in enumerate(train_features.columns)
        }
        return self.baseline_importance

    def _resolve_drifted_samples(
        self,
        incoming_drifted_df: Optional[pd.DataFrame] = None,
        drifted_features: Optional[List[str]] = None,
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        if incoming_drifted_df is not None and not incoming_drifted_df.empty:
            return incoming_drifted_df

        if self.reference_path and os.path.exists(self.reference_path) and drifted_features:
            with open(self.reference_path, "rb") as f:
                reference = pickle.load(f)
            means = reference.get("feature_means", {})
            stds = reference.get("feature_stds", {})

            mask = np.zeros(len(self.test_df), dtype=bool)
            for feat in drifted_features:
                if feat in self.test_df.columns and feat in means:
                    std = float(stds.get(feat, 0.0))
                    if std <= 1e-9:
                        continue
                    z = (self.test_df[feat] - float(means[feat])) / std
                    mask = mask | (np.abs(z) >= z_threshold)
            candidate = self.test_df[mask]
            if not candidate.empty:
                return candidate

        return self.test_df

    def run_rca(
        self,
        drifted_features: List[str],
        incoming_drifted_df: Optional[pd.DataFrame] = None,
        shift_threshold_ratio: float = 1.5,
    ) -> Dict:
        if not self.baseline_importance:
            self.build_baseline()

        drifted_df = self._resolve_drifted_samples(
            incoming_drifted_df=incoming_drifted_df,
            drifted_features=drifted_features,
        )
        drifted_features_df = self._prepare_features(drifted_df)
        train_features = self._prepare_features(self.train_df)

        explainer = _build_explainer(self.model, train_features)
        drifted_shap_values = explainer(drifted_features_df)
        drifted_positive_shap = _extract_positive_class_shap_values(drifted_shap_values)
        drifted_mean_abs = np.mean(np.abs(drifted_positive_shap), axis=0)

        feature_comparison = []
        for idx, feature in enumerate(drifted_features_df.columns):
            baseline_value = float(self.baseline_importance.get(feature, 0.0))
            current_value = float(drifted_mean_abs[idx])
            shift_delta = current_value - baseline_value
            shift_ratio = (current_value / baseline_value) if baseline_value > 1e-12 else None
            is_rogue = (
                shift_ratio is not None and shift_ratio >= shift_threshold_ratio and feature in drifted_features
            )
            feature_comparison.append(
                {
                    "feature": feature,
                    "baseline_mean_abs_shap": _to_serializable_number(baseline_value),
                    "drifted_mean_abs_shap": _to_serializable_number(current_value),
                    "shift_delta": _to_serializable_number(shift_delta),
                    "shift_ratio": _to_serializable_number(shift_ratio),
                    "is_rogue": bool(is_rogue),
                }
            )

        rogue_features = [item["feature"] for item in feature_comparison if item["is_rogue"]]
        explanation = (
            f"Drift was detected in {len(drifted_features)} feature(s). "
            f"RCA compared SHAP influence on baseline training data versus drifted batch samples. "
            f"{len(rogue_features)} rogue feature(s) showed a significant increase in predictive influence: "
            f"{', '.join(rogue_features) if rogue_features else 'none'}."
        )

        report = {
            "report_type": "drift_root_cause_analysis",
            "generated_at_utc": _utc_now_iso(),
            "input_artifacts": {
                "model_path": self.model_path,
                "train_csv_path": self.train_csv_path,
                "test_csv_path": self.test_csv_path,
                "reference_path": self.reference_path,
            },
            "drifted_features": drifted_features,
            "rogue_features": rogue_features,
            "num_drifted_samples_analyzed": int(len(drifted_features_df)),
            "baseline_feature_importance": self.baseline_importance,
            "feature_comparison": feature_comparison,
            "plain_english_explanation": explanation,
        }
        return report


def generate_rca_report(
    model_path: str,
    train_csv_path: str,
    test_csv_path: str,
    drifted_features: List[str],
    reference_path: Optional[str] = None,
    drifted_batch_records: Optional[List[Dict]] = None,
    shift_threshold_ratio: float = 1.5,
) -> Dict:
    explainer = DriftRCAExplainer(
        model_path=model_path,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        reference_path=reference_path,
    )
    explainer.load()
    incoming_df = pd.DataFrame(drifted_batch_records) if drifted_batch_records else None
    return explainer.run_rca(
        drifted_features=drifted_features,
        incoming_drifted_df=incoming_df,
        shift_threshold_ratio=shift_threshold_ratio,
    )


def main():
    parser = argparse.ArgumentParser(description="Run SHAP-based RCA for drifted churn features.")
    parser.add_argument("--model-path", default="/data/churn-model/churn_model.pkl")
    parser.add_argument("--train-csv-path", default="/data/churn-model/train.csv")
    parser.add_argument("--test-csv-path", default="/data/churn-model/test.csv")
    parser.add_argument("--reference-path", default="/data/churn-model/reference_distribution.pkl")
    parser.add_argument("--drifted-features", required=True, help="Comma-separated drifted feature names")
    parser.add_argument("--output-path", default="/tmp/rca_report.json")
    parser.add_argument("--shift-threshold-ratio", type=float, default=1.5)
    args = parser.parse_args()

    drifted_features = [x.strip() for x in args.drifted_features.split(",") if x.strip()]
    report = generate_rca_report(
        model_path=args.model_path,
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        drifted_features=drifted_features,
        reference_path=args.reference_path,
        shift_threshold_ratio=args.shift_threshold_ratio,
    )
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"status": "success", "output_path": args.output_path}))


if __name__ == "__main__":
    main()
