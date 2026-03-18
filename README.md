# SPE Final Project – Microservice Pipeline

This project implements a complete CI/CD pipeline for a churn prediction system using four microservices. It covers the full machine learning lifecycle, from automated training to real-time inference and data drift monitoring.

---

## Microservices Overview

- **Model Training (Port 5001)**  
  Trains and saves the churn prediction model.

- **Model Serving (Port 5002)**  
  Serves predictions via a REST API.

- **Drift Detection (Port 5003)**  
  Monitors incoming data drift against the training distribution.

- **Data Ingestion (Port 5000)**  
  Ingests customer data and coordinates calls to serving and drift detection.

---

## Architecture & Tech Stack

- Microservices Framework: Flask  
- Containerization: Docker  
- Orchestration: Kubernetes  
- CI/CD: Jenkins  
- Infrastructure as Code: Ansible  
- Security: Ansible Vault (AES-256 encryption)  
- Storage: Kubernetes PV/PVC for shared model storage  
- Networking: Host Networking (for local testing)

---

## Project Structure

```
SPEFinalProject/
├── data_ingestion/     
├── model_training/     
├── model_serving/      
├── drift_detection/    
├── kubernetes/         
├── ansible/            
│   ├── vars/
│   │   └── secrets.yml 
│   └── site.yml        
├── docker-compose.test.yml
└── Jenkinsfile
```

---

## Secure Storage (Ansible Vault)

All sensitive credentials are encrypted using AES-256 via Ansible Vault.

### Edit Secrets

```bash
ansible-vault edit ansible/vars/secrets.yml
```

### Run Provisioning

```bash
ansible-playbook -i ansible/inventory.ini ansible/site.yml \
--ask-vault-pass --ask-become-pass
```

---

## Setup & Installation

### Build Docker Images

```bash
docker build -t kirtinigam003/model_training:latest ./model_training
docker build -t kirtinigam003/model_serving:latest ./model_serving
docker build -t kirtinigam003/drift_detection:latest ./drift_detection
docker build -t kirtinigam003/data_ingestion:latest ./data_ingestion
```

---

## Run Locally (Docker Compose)

```bash
export HOST_IP=$(hostname -I | awk '{print $1}')
docker compose -f docker-compose.test.yml up --build
```

---

## API Testing

### Step 1: Train Model

```bash
curl -X POST http://localhost:5001/train \
-H "Content-Type: application/json" \
-d '[{"Age":45,"Tenure":5,"Balance":2000,"Churn":"No"},
{"Age":30,"Tenure":2,"Balance":500,"Churn":"Yes"}]'
```

### Step 2: Test Ingestion Pipeline

```bash
curl -i -X POST http://localhost:5000/ingest \
-H "Content-Type: application/json" \
-d '[{"customerID":"C001","Age":35,"Tenure":3,"Balance":2500,"Churn":"No"}]'
```

---

## Kubernetes Deployment

### Setup Storage

```bash
kubectl apply -f kubernetes/pv.yaml
kubectl apply -f kubernetes/pvc.yaml
```

### Deploy Services

```bash
kubectl apply -f kubernetes/deployment/
kubectl apply -f kubernetes/service/
```

---

## CI/CD Pipeline Flow

1. **Checkout** – Pull latest code from GitHub  
2. **Environment Setup** – Provision VM using Ansible  
3. **Security** – Fetch Ansible Vault credentials  
4. **Build & Push** – Build Docker images and push to Docker Hub  
5. **Integration Test** – Run containers, train model, and validate pipeline  
6. **Deploy** – Update Kubernetes cluster with latest images  

---

## Notes

- Ensure at least two classes (`Churn: Yes` and `No`) during training.
- PVC is used to share model files between training and serving services.
- Host networking is used locally for easier communication.

---
