pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        IMAGE_REPOSITORY = 'docker.io/bapps/cicd'
        K8S_NAMESPACE = 'bapps'
        K8S_DEPLOYMENT = 'bapps-backend'
        K8S_CONTAINER = 'api'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.IMAGE_TAG = sh(script: 'git rev-parse --short=12 HEAD', returnStdout: true).trim()
                    env.IMAGE = "${env.IMAGE_REPOSITORY}:${env.IMAGE_TAG}"
                }
            }
        }

        stage('Build and test') {
            steps {
                sh '''
                    python3 -m venv .ci-venv
                    . .ci-venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    python -m compileall -q app
                    python smoke_test.py
                '''
            }
        }

        stage('Build image') {
            steps {
                sh 'docker build --pull -t "$IMAGE" .'
            }
        }

        stage('Push image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_TOKEN')]) {
                    sh '''
                        set +x
                        printf '%s' "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USER" --password-stdin
                        docker push "$IMAGE"
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Minikube') {
            steps {
                sh '''
                    kubectl apply -f k8s/namespace.yaml
                    kubectl apply -n "$K8S_NAMESPACE" -f k8s/pvc.yaml
                    kubectl apply -n "$K8S_NAMESPACE" -f k8s/deployment.yaml
                    kubectl apply -n "$K8S_NAMESPACE" -f k8s/service.yaml
                    kubectl set image "deployment/$K8S_DEPLOYMENT" "$K8S_CONTAINER=$IMAGE" -n "$K8S_NAMESPACE"
                    kubectl rollout status "deployment/$K8S_DEPLOYMENT" -n "$K8S_NAMESPACE" --timeout=180s
                '''
            }
        }

        stage('Verify deployment') {
            steps {
                sh '''
                    kubectl get deployment,pods,service -n "$K8S_NAMESPACE" -o wide
                    kubectl get deployment "$K8S_DEPLOYMENT" -n "$K8S_NAMESPACE" -o jsonpath='{.status.readyReplicas}' | grep -q '^1$'
                    kubectl get pods -n "$K8S_NAMESPACE" -l app=bapps-backend --field-selector=status.phase=Running | grep -q bapps-backend
                    kubectl get service bapps-backend -n "$K8S_NAMESPACE"
                    kubectl port-forward service/bapps-backend 18000:8000 -n "$K8S_NAMESPACE" >/tmp/bapps-port-forward.log 2>&1 &
                    PORT_FORWARD_PID=$!
                    trap 'kill "$PORT_FORWARD_PID" 2>/dev/null || true' EXIT
                    for attempt in $(seq 1 20); do
                        if curl --fail --silent http://127.0.0.1:18000/api/health; then
                            echo
                            break
                        fi
                        sleep 2
                    done
                    curl --fail --silent http://127.0.0.1:18000/api/health >/dev/null
                    echo "Verify in browser with: minikube service bapps-backend -n $K8S_NAMESPACE --url"
                '''
            }
        }
    }

    post {
        always {
            sh 'docker image rm -f "$IMAGE" 2>/dev/null || true'
            cleanWs()
        }
    }
}
