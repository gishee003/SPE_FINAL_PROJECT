pipeline {
    agent any
    environment {
        // Use the ID you set in Jenkins Credentials for the kubeconfig file
        HOST_IP = sh(script: "hostname -I | awk '{print \$1}'", returnStdout: true).trim()
        KUBECONFIG = credentials('kubeconfig') 
    }
    stages {
        stage('Environment Setup') {
            steps {
                withCredentials([string(credentialsId: 'ansible-vault-pass', variable: 'VAULT_PW')]) {
                    // Create a temporary file for the password
                    sh 'echo $VAULT_PW > .vault_pass'
                    
                    // Run Ansible using the password file
                    sh 'ansible-playbook -i ansible/inventory.ini ansible/site.yml --vault-password-file .vault_pass'
                    
                    // Clean up the password file immediately
                    sh 'rm .vault_pass'
                }
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r data_ingestion/requirements.txt
                    python3 -m unittest data_ingestion/tests/test_ingestion.py
                    python3 -m unittest drift_detection/tests/test_drift_detection.py
                    python3 -m unittest model_serving/tests/test_serving.py
                    python3 -m unittest model_training/tests/test_training.py
                '''
            }
        }

        stage('Integration Testing') {
            options {
                timeout(time: 5, unit: 'MINUTES')
            }
            steps {
                script {
                    def hostIp = sh(script: "hostname -I | awk '{print \$1}'", returnStdout: true).trim()
                    echo "Detected Host IP for Integration: ${hostIp}"

                    sh 'docker compose -f docker-compose.test.yml down --remove-orphans'
                    sh 'docker compose -f docker-compose.test.yml up --build -d'
                    
                    echo "Waiting for Flask services to initialize..."
                    sleep 15

                    // Bootstrap Training
                    echo "Training baseline model..."
                    def trainStatus = sh(script: """
                        curl -s -X POST http://${hostIp}:5001/train \
                        -H "Content-Type: application/json" \
                        -d '[{"Age": 30, "Tenure": 5, "Balance": 1000, "Churn": "No"}, {"Age": 45, "Tenure": 1, "Balance": 8000, "Churn": "Yes"}]'
                    """, returnStdout: true).trim()
                    
                    if (!trainStatus.contains("success")) {
                        error "Model Training failed: ${trainStatus}"
                    }

                    // Full Pipeline Test
                    def response = sh(script: "curl -s -X POST http://${hostIp}:5000/ingest -H 'Content-Type: application/json' -d '[{\"customerID\": \"123\", \"Age\": 35, \"Tenure\": 3, \"Balance\": 2500, \"Churn\": \"No\"}]'", returnStdout: true).trim()
                    echo "Full Pipeline Response: ${response}"

                    if (response.contains("error") || !response.contains("ingested")) {
                        sh 'docker compose -f docker-compose.test.yml logs'
                        error "Integration Test Failed. Response: ${response}"
                    }
                }
            }
            post {
                always {
                    sh 'docker compose -f docker-compose.test.yml down'
                }
            }
        }

        stage('Build & Push Images') {
            parallel {
                stage('Ingestion') {
                    steps {
                        dir('data_ingestion') {
                            sh 'docker build -t kirtinigam003/data_ingestion:latest -f Dockerfile.ingest .'
                            sh 'docker push kirtinigam003/data_ingestion:latest'
                        }
                    }
                }
                stage('Training') {
                    steps {
                        dir('model_training') {
                            sh 'docker build -t kirtinigam003/model_training:latest -f Dockerfile.training .'
                            sh 'docker push kirtinigam003/model_training:latest'
                        }
                    }
                }
                stage('Serving') {
                    steps {
                        dir('model_serving') {
                            sh 'docker build -t kirtinigam003/model_serving:latest -f Dockerfile.serving .'
                            sh 'docker push kirtinigam003/model_serving:latest'
                        }
                    }
                }
                stage('Drift') {
                    steps {
                        dir('drift_detection') {
                            sh 'docker build -t kirtinigam003/drift_detection:latest -f Dockerfile.drift .'
                            sh 'docker push kirtinigam003/drift_detection:latest'
                        }
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh '''
                    export MINIKUBE_HOME=/home/kirtinigam003
                    export KUBECONFIG=/home/kirtinigam003/.kube/config
                        
                    echo "Starting Deployment..."

                        # 1. Check if Minikube is actually Running, ignoring stale warnings
                        if ! minikube status | grep -q "Running"; then
                            echo "❌ ERROR: Minikube is not running. Run 'minikube start' on the host."
                            exit 1
                        fi

                        # 2. Deploy using --validate=false to skip the network check that keeps failing
                        # This bypasses the 'connection refused' error on the openapi schema
                        echo "Applying Kubernetes manifests..."
                        kubectl apply -f kubernetes/pv.yaml --validate=false
                        kubectl apply -f kubernetes/pvc.yaml --validate=false
                        
                        kubectl apply -f kubernetes/deployment/ --validate=false
                        kubectl apply -f kubernetes/service/ --validate=false

                        # 3. Show current status
                        echo "✅ Deployment commands sent successfully."
                        kubectl get pods
                '''
            }
        }
    }
}