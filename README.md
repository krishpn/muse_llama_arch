cd apps/nkepsx/backend

`uv pip install -e .`
`uv run pytest tests/ -v`
`uv sync`


# 1. Backend & Container Build Cycle

`docker images` 

Remove old image locally

`docker rmi -f nkepsx-backend:v4`

Rebuild fresh without cache

`docker build --no-cache -t nkepsx-backend:v4 .`

Save and import into K3s containerd

`docker save nkepsx-backend:v4 -o /tmp/backend-v4.tar`

`sudo k3s ctr images import /tmp/backend-v4.tar`

Restart deployment to pick up the new image

`kubectl rollout restart deployment/nkepsx-orchestrator`


# 2. Kubernetes Cluster & Pod Diagnostics

Check status of all pods

`kubectl get pods`

Check status of services & NodePorts

`kubectl get svc`

View live container logs (tail last 50 lines)

`kubectl logs deployment/nkepsx-orchestrator --tail=50 -f`

Describe pod for startup/pull errors

`kubectl describe pod -l app=nkepsx-orchestrator`

List images currently imported into K3s containerd

`sudo k3s ctr images list | grep nkepsx`

Open an interactive shell inside the running orchestrator pod

`kubectl exec -it deployment/nkepsx-orchestrator -- /bin/bash`

Run a quick check on installed Python packages inside the pod

`kubectl exec -it deployment/nkepsx-orchestrator -- pip list`

Test internal DNS resolution from inside the pod (e.g., reaching MongoDB service)

`kubectl exec -it deployment/nkepsx-orchestrator -- python -c "import socket; print(socket.gethostbyname('nkepsx-mongodb-svc'))"`

Verify environment variables set inside the pod container

`kubectl exec -it deployment/nkepsx-orchestrator -- env`

# 3. API & Endpoint Testing

Test health check root

`curl -i http://localhost:30080/`

Test dataset collection listing

`curl -i http://localhost:30080/api/datasets`

Test CRSP analytics summary route

`curl -i http://localhost:30080/api/v1/analytics/crsp-summary`
