# SPE Final Project – Microservice Pipeline

This project implements a complete CI/CD pipeline for a churn prediction system using **four microservices**. The architecture handles the full machine learning lifecycle, from training to real-time inference and monitoring.

---

## 🛠 Microservices Overview

* **Model Training:** Trains and saves the churn prediction model.
* **Model Serving:** Serves predictions via a REST API.
* **Drift Detection:** Monitors incoming data drift against the training distribution.
* **Data Ingestion:** Ingests customer data and coordinates calls to serving and drift detection.

All services are containerized with **Docker**, orchestrated via **Kubernetes**, and automated through a **Jenkins** CI/CD pipeline.

---

## 📂 Project Structure

```text
SPEFinalProject/
├── data_ingestion/
├── model_training/
├── model_serving/
├── drift_detection/
├── kubernetes/
│   ├── deployment/
│   ├── service/
│   ├── pv.yaml
│   └── pvc.yaml
└── Jenkinsfile
```

## Setup & Installation

### 1. Build Docker Images
From the project root, build the images for each service:

```
docker build -t kirtinigam003/model_training:latest ./model_training
docker build -t kirtinigam003/model_serving:latest ./model_serving
docker build -t kirtinigam003/drift_detection:latest ./drift_detection
docker build -t kirtinigam003/data_ingestion:latest ./data_ingestion
```

### 2. Run Locally (Using Host Network)
Open four separate terminals and execute the following:

```
#1. Model Training
docker run -it --rm --network host \
  -v $(pwd)/data:/data/churn-model \
  kirtinigam003/model_training:latest

#2. Model Serving
docker run -it --rm --network host \
  -v $(pwd)/data:/data/churn-model \
  kirtinigam003/model_serving:latest

#3. Drift Detection
docker run -it --rm --network host \
  -v $(pwd)/data:/data/churn-model \
  -e TRAINING_URL=http://localhost:5001/train \
  kirtinigam003/drift_detection:latest

#4. Data Ingestion
docker run -it --rm --network host \
  -e SERVING_URL=http://localhost:5002/predict \
  -e DRIFT_URL=http://localhost:5003/drift \
  kirtinigam003/data_ingestion:latest
```

## API Endpoints & Test Commands

### Train the Model
```
curl -X POST http://localhost:5001/train \
  -H "Content-Type: application/json" \
  -d '[{"Age":45,"Tenure":5,"Balance":2000,"Churn":"No"}, {"Age":30,"Tenure":2,"Balance":500,"Churn":"Yes"}]'
```

### Get a Prediction
```
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '[{"customerID":"C001","Age":45,"Tenure":5,"Balance":2000}]'
```

### Check Drift
```
curl -X POST http://localhost:5003/drift \
  -H "Content-Type: application/json" \
  -d '[{"Age":45,"Tenure":5,"Balance":2000,"Churn":"No"}]'
```

### Ingest Data
```
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '[{"customerID":"C001","Age":45,"Tenure":5,"Balance":2000,"Churn":"No"}]'
```

## Kubernetes Depolyment
Ensure your images are pushed to Docker Hub, then apply the manifests:

```
# 1. Setup Storage
kubectl apply -f kubernetes/pv.yaml
kubectl apply -f kubernetes/pvc.yaml

# 2. Deploy Services
kubectl apply -f kubernetes/deployment/data_ingestion_deployment.yaml
kubectl apply -f kubernetes/deployment/model_training_deployment.yaml
kubectl apply -f kubernetes/deployment/model_serving_deployment.yaml
kubectl apply -f kubernetes/deployment/drift_detection_deployment.yaml

# 3. Apply Services
kubectl apply -f kubernetes/service/data_ingestion_service.yaml
kubectl apply -f kubernetes/service/model_training_service.yaml
kubectl apply -f kubernetes/service/model_serving_service.yaml
kubectl apply -f kubernetes/service/drift_detection_service.yaml

```

### Key Notes

PVC: Used to share the .pkl model between training and serving.

Ports: Ingestion (5000), Training (5001), Serving (5002), Drift (5003).

CI/CD: Jenkinsfile automates the build/push/deploy cycle.

### Pipeline Flow

```
flowchart LR
    A[Model Training] --> B[Model Serving]
    A --> C[Drift Detection]
    B --> D[Data Ingestion]
    C --> D
    D -->|Final Response| User
```
