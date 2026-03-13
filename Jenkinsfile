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
                    python3 -m unittest model_serving/tests/test_serving.py
                    python3 -m unittest model_training/tests/test_training.py
                '''
            }
        }
        stage('Integration Testing') {
            options {
                // Prevents the job from hanging forever if something goes wrong
                timeout(time: 5, unit: 'MINUTES')
            }
            steps {
                script {
                    // 1. Identify the Host IP (Required for Host Networking mode)
                    // This gets the primary IP of the Ubuntu runner
                    def hostIp = sh(script: "hostname -I | awk '{print \$1}'", returnStdout: true).trim()
                    echo "Detected Host IP for Integration: ${hostIp}"

                    // 2. Start the Stack
                    // We use --build to ensure the 'global model' fix is included
                    sh 'docker compose -f docker-compose.test.yml down --remove-orphans'
                    sh 'docker compose -f docker-compose.test.yml up --build -d'
                    
                    echo "Waiting for Flask services to initialize..."
                    sleep 15

                    // 3. Bootstrap: Train the model
                    // We MUST do this first so the 'serving' service has a model to load
                    echo "Training baseline model..."
                    def trainStatus = sh(script: """
                        curl -s -X POST http://${hostIp}:5001/train \
                        -H "Content-Type: application/json" \
                        -d '[
                            {"Age": 30, "Tenure": 5, "Balance": 1000, "Churn": "No"},
                            {"Age": 45, "Tenure": 1, "Balance": 8000, "Churn": "Yes"}
                        ]'
                    """, returnStdout: true).trim()
                    
                    echo "Training Response: ${trainStatus}"
                    if (!trainStatus.contains("success")) {
                        error "Model Training failed, cannot proceed with Integration Test."
                    }

                    // 4. Integration Test: Full Data Ingestion
                    echo "Running Full Pipeline Test..."
                    def response = sh(script: """
                        curl -s -X POST http://${hostIp}:5000/ingest \
                        -H "Content-Type: application/json" \
                        -d '[{"customerID": "123", "Age": 35, "Tenure": 3, "Balance": 2500, "Churn": "No"}]'
                    """, returnStdout: true).trim()

                    echo "Full Pipeline Response: ${response}"

                    // 5. Validation Logic
                    // We check for 'ingested' AND ensure no Python 'error' exists in the nested responses
                    if (response.contains("ingested") && !response.contains("error")) {
                        echo "SUCCESS: Ingest, Serving, and Drift services are all synchronized."
                    } else {
                        // If it fails, print logs to Jenkins console for debugging
                        sh 'docker compose -f docker-compose.test.yml logs'
                        error "Integration Test Failed. Response received: ${response}"
                    }
                }
            }
            post {
                always {
                    // Clean up to free up ports 5000-5004 for the next build
                    sh 'docker compose -f docker-compose.test.yml down'
                }
            }
        }
        stage('Build Data Ingestion') {
            steps {
                dir('data_ingestion') {
                    sh 'docker build -t kirtinigam003/data_ingestion:latest -f Dockerfile.ingest .'
                }
            }
        }
        stage('Build Model Training') {
            steps {
                dir('model_training') {
                    sh 'docker build -t kirtinigam003/model_training:latest -f Dockerfile.training .'
                }
            }
        }
        stage('Build Model Serving') {
            steps {
                dir('model_serving') {
                    sh 'docker build -t kirtinigam003/model_serving:latest -f Dockerfile.serving .'
                }
            }
        }
        stage('Build Drift Detection') {
            steps {
                dir('drift_detection') {
                    sh 'docker build -t kirtinigam003/drift_detection:latest -f Dockerfile.drift .'
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
