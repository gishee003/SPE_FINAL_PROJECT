pipeline {
    agent any
    environment {
        HOST_IP = sh(script: "hostname -I | awk '{print \$1}'", returnStdout: true).trim()
        KUBECONFIG = credentials('kubeconfig') 
    }
    stages {
        stage('Environment Setup') {
            steps {
                withCredentials([string(credentialsId: 'ansible-vault-pass', variable: 'VAULT_PW')]) {
                    sh 'echo $VAULT_PW > .vault_pass'
                    
                    sh 'ansible-playbook -i ansible/inventory.ini ansible/site.yml --vault-password-file .vault_pass'
                    
                    sh 'rm .vault_pass'
                }
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
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

                    echo "Training baseline model..."
                    def trainStatus = sh(script: """
                        curl -s -X POST http://${hostIp}:5001/train \
                        -H "Content-Type: application/json" \
                        -d '[{
                            "CustomerId": 1,
                            "CreditScore": 600,
                            "Geography": "France",
                            "Gender": "Male",
                            "Age": 30,
                            "Tenure": 5,
                            "Balance": 1000.0,
                            "NumOfProducts": 1,
                            "HasCrCard": 1,
                            "IsActiveMember": 1,
                            "EstimatedSalary": 40000.0,
                            "Exited": 0
                        },
                        {
                            "CustomerId": 2,
                            "CreditScore": 650,
                            "Geography": "Germany",
                            "Gender": "Female",
                            "Age": 45,
                            "Tenure": 1,
                            "Balance": 8000.0,
                            "NumOfProducts": 2,
                            "HasCrCard": 0,
                            "IsActiveMember": 0,
                            "EstimatedSalary": 50000.0,
                            "Exited": 1
                        }]'
                    """, returnStdout: true).trim()
                    
                    if (!trainStatus.contains("success")) {
                        error "Model Training failed: ${trainStatus}"
                    }

                    
                    echo "Waiting for training artifacts..."
                    sleep 10

                    def response = sh(script: """
                        curl -s -X POST http://${hostIp}:5000/ingest \
                        -H 'Content-Type: application/json' \
                        -d '[{
                            "CustomerId": 123,
                            "CreditScore": 700,
                            "Geography": "Spain",
                            "Gender": "Female",
                            "Age": 35,
                            "Tenure": 3,
                            "Balance": 2500.0,
                            "NumOfProducts": 1,
                            "HasCrCard": 1,
                            "IsActiveMember": 1,
                            "EstimatedSalary": 45000.0,
                            "Exited": 0
                        },
                        {
                            "CustomerId": 1234,
                            "CreditScore": 600,
                            "Geography": "Germany",
                            "Gender": "Female",
                            "Age": 35,
                            "Tenure": 4,
                            "Balance": 4000.0,
                            "NumOfProducts": 1,
                            "HasCrCard": 1,
                            "IsActiveMember": 1,
                            "EstimatedSalary": 60000.0,
                            "Exited": 1
                        }]'
                    """, returnStdout: true).trim()
                    echo "Full Pipeline Response: ${response}"

                    if (response.contains("error") || !response.contains("ingested")) {
                        sh 'docker compose -f docker-compose.test.yml logs'
                        error "Integration Test Failed. Response: ${response}"
                    }
                }
            }
            post {
                always {
                    sh 'docker compose -f docker-compose.test.yml ps -a'
                    sh 'docker compose -f docker-compose.test.yml logs training'
                    sh 'docker compose -f docker-compose.test.yml down || true'
                }
            }
        }

	stage('Docker Login') {
		steps {
    			withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
      				sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
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
            environment {
                KUBECONFIG = '/var/lib/jenkins/.kube/config'
            }
            steps {
                sh '''
                    echo "Starting Deployment..."
                    sed -i 's|/home/kirti/.minikube|/var/lib/jenkins/.minikube|g' /var/lib/jenkins/.kube/config
                    
                    kubectl cluster-info

                    echo "Applying Kubernetes manifests..."
                    kubectl apply -f kubernetes/pv.yaml --validate=false
                    kubectl apply -f kubernetes/pvc.yaml --validate=false
                    
                    kubectl apply -f kubernetes/deployment/ --validate=false
                    kubectl apply -f kubernetes/service/ --validate=false

                    kubectl apply -f kubernetes/hpa.yaml --validate=false

                    echo "✅ Deployment commands sent successfully."
                    kubectl get pods
                '''
            }
        }

        stage('ELK Dashboard Setup') {
            environment {
                MINIKUBE_HOME='/home/kirti'
                KUBECONFIG = '/var/lib/jenkins/.kube/config'
            }
            steps {
                sh '''
                    echo "Setting up ELK + Kibana dashboard..."

                    kubectl apply -f kubernetes/elk/elasticsearch.yaml --validate=false
                    kubectl apply -f kubernetes/elk/filebeat.yaml --validate=false
                    kubectl apply -f kubernetes/elk/kibana.yaml --validate=false
                    kubectl apply -f kubernetes/elk/project-logs-es-mappings-configmap.yaml --validate=false
                    kubectl apply -f kubernetes/elk/kibana-dashboard-config.yaml --validate=false

                    kubectl rollout status deployment/elasticsearch --timeout=300s
                    kubectl rollout status deployment/kibana --timeout=300s

                    kubectl delete job kibana-setup --ignore-not-found=true
                    kubectl apply -f kubernetes/elk/kibana-setup.yaml --validate=false
                    if ! kubectl wait --for=condition=complete job/kibana-setup --timeout=420s; then
                        echo "kibana-setup did not complete in time. Collecting diagnostics..."
                        kubectl describe job kibana-setup || true
                        kubectl logs job/kibana-setup --all-containers=true --tail=300 || true
                        kubectl get pods -l app=kibana-setup -o wide || true
                        kubectl get pods -l app=filebeat -o wide || true
                        kubectl logs -l app=filebeat --tail=200 || true
                        exit 1
                    fi
                    kubectl logs job/kibana-setup --all-containers=true --tail=300

                    echo "✅ ELK dashboard setup complete."
                '''
            }
        }
        
    }
}
