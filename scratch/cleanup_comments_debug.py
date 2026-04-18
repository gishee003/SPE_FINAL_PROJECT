from pathlib import Path
import re


ROOT = Path("/home/kirti/SPE/SPE_FINAL_PROJECT")

TARGETS = [
    "Jenkinsfile",
    ".gitignore",
]

EXTS = {".py", ".yaml", ".yml", ".sh", ".ini"}


def should_process(path: Path) -> bool:
    if ".git" in path.parts or "venv" in path.parts:
        return False
    if "agent-transcripts" in path.parts or "mcps" in path.parts:
        return False
    if path.name in TARGETS:
        return True
    if path.suffix in EXTS:
        return True
    if path.name.startswith("Dockerfile"):
        return True
    return False


def is_comment_only(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("#!"):
        return False
    return stripped.startswith("#") or stripped.startswith("//")


def is_debug_line(line: str) -> bool:
    stripped = line.strip()
    if re.search(r"\bprint\(", stripped) and "logging" not in stripped:
        return True
    return False


changed = 0

for path in ROOT.rglob("*"):
    if not path.is_file() or not should_process(path):
        continue
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        continue
    lines = raw.splitlines()
    out = []
    for line in lines:
        if is_comment_only(line):
            continue
        if is_debug_line(line):
            continue
        out.append(line)
    new_raw = "\n".join(out) + ("\n" if raw.endswith("\n") else "")
    if new_raw != raw:
        path.write_text(new_raw, encoding="utf-8")
        changed += 1

