pipeline {
    agent any
    environment {
        KUBECONFIG = credentials('kubeconfig')  // Jenkins secret file with kubeconfig
    }
    stages {

        stage('Run Unit Tests') {
            steps {
                sh '''
                    # 1. Setup virtual environment (recommended)
                    python3 -m venv venv
                    . venv/bin/activate

                    # 2. Install dependencies
                    pip install -r data_ingestion/requirements.txt
                    
                    # 3. Run the tests from the root
                    # We use the -m flag just like you did in the terminal
                    python3 -m unittest data_ingestion/tests/test_ingestion.py
                    python3 -m unittest drift_detection/tests/test_drift_detection.py
                '''
            }
        }
        stage('Build Data Ingestion') {
            steps {
                dir('data_ingestion') {
                    sh 'docker build -t kirtinigam003/data_ingestion:latest .'
                }
            }
        }
        stage('Build Model Training') {
            steps {
                dir('model_training') {
                    sh 'docker build -t kirtinigam003/model_training:latest .'
                }
            }
        }
        stage('Build Model Serving') {
            steps {
                dir('model_serving') {
                    sh 'docker build -t kirtinigam003/model_serving:latest .'
                }
            }
        }
        stage('Build Drift Detection') {
            steps {
                dir('drift_detection') {
                    sh 'docker build -t kirtinigam003/drift_detection:latest .'
                }
            }
        }
        stage('Push Images') {
            steps {
                sh 'docker push kirtinigam003/data_ingestion:latest'
                sh 'docker push kirtinigam003/model_training:latest'
                sh 'docker push kirtinigam003/model_serving:latest'
                sh 'docker push kirtinigam003/drift_detection:latest'
            }
        }
        stage('Deploy to Kind') {
            steps {
                sh '''
                echo "Using kubeconfig at $KUBECONFIG"
                kubectl config view --minify
                kubectl get nodes
                '''
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        kubectl config view
                        kubectl apply -f kubernetes/pv.yaml
                        kubectl apply -f kubernetes/pvc.yaml
                        kubectl apply -f kubernetes/deployment/data_ingestion_deployment.yaml
                        kubectl apply -f kubernetes/deployment/model_training_deployment.yaml
                        kubectl apply -f kubernetes/deployment/model_serving_deployment.yaml
                        kubectl apply -f kubernetes/deployment/drift_detection_deployment.yaml
                        kubectl apply -f kubernetes/service/data_ingestion_service.yaml
                        kubectl apply -f kubernetes/service/model_training_service.yaml
                        kubectl apply -f kubernetes/service/model_serving_service.yaml
                        kubectl apply -f kubernetes/service/drift_detection_service.yaml
                    '''
                }
            }
        }
    }
}
