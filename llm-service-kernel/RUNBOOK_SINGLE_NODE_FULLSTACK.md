
# Single-Node Fullstack Runbook

## Purpose

This runbook is the canonical guide for running the **single-node Kubernetes fullstack** version of the service.

It covers:

- what the system does
- which files are the official ones
- how to build and deploy it
- how to ingest tenant PDFs
- how to test RAG and semantic cache
- useful Kubernetes commands for debugging

This is the **official path**. Legacy files and older host-mount workflows are not the main flow anymore.

---

# 1. System overview

## Target pipeline

```text
Client
→ FastAPI (llm-service-kernel, in-cluster)
→ Orchestrator
   → Semantic Cache
   → RAG
→ llm-d gateway
→ vLLM
→ model
→ response
````

## Current single-node architecture

This setup runs on **one Minikube node**.

The main runtime components are:

* **FastAPI**: request entrypoint and orchestration layer
* **MongoDB**: semantic cache scalar metadata/indexes
* **Milvus**: semantic cache vectors and RAG vectors
* **SeaweedFS**: object store for PDFs and chunk payloads
* **llm-d / vLLM**: generation backend
* **Tenant ingest job**: parses and indexes tenant PDFs

## Data responsibilities

### FastAPI

Handles:

* chat requests
* routing between semantic cache / RAG / plain backend
* health reporting
* interaction with llm-d

### MongoDB

Handles:

* semantic cache document metadata
* TTL cleanup for semantic cache entries

### Milvus

Handles:

* semantic cache embeddings
* RAG embeddings for document chunks

### SeaweedFS

Handles:

* original PDF storage
* chunk JSON payload storage

### `rag_store_tenants`

Currently used for:

* per-tenant manifest storage
* `manifest.json` files that tell the service which tenant knowledge base version is active

This still works and is part of the current deployment, but it is a transitional design.

---

# 2. Official files and what they do

## Main build/deploy files

### `Dockerfile.service`

Main FastAPI image build file.

It packages:

* the service code
* runtime assets
* vendored offline dependencies
* scripts needed for the official flow

### `scripts/prepare_fastapi_runtime_assets.sh`

Builds the local `fastapi_runtime_assets/` bundle used by the FastAPI image.

It collects:

* `all-MiniLM-L6-v2`
* `bge-base-en-v1.5`
* tokenizer files from `Qwen2.5-0.5B-Instruct`
* seed `rag_store_tenants`

### `scripts/vendor_local_pypdf.sh`

Copies the locally installed `pypdf` package into `vendor/` so the image can use it **offline**.

This is needed because the environment is offline and the ingestion script requires PDF parsing.

### `scripts/deploy_fastapi_fullstack.sh`

Official fullstack deployment entrypoint.

It:

* applies storage manifests
* applies FastAPI config/secret/service
* creates required PVCs
* runs the bootstrap job
* deploys/restarts the FastAPI deployment

---

## Kubernetes files

### `deploy/k8s-storage/mongo.yaml`

Deploys MongoDB and its PVC.

### `deploy/k8s-storage/milvus.yaml`

Deploys Milvus standalone, etcd, MinIO, and their PVCs.

### `deploy/k8s-storage/seaweedfs.yaml`

Deploys SeaweedFS master, volume, filer, S3 gateway, and their PVCs/config.

### `deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml`

Main non-secret runtime config for FastAPI.

Contains things like:

* backend mode
* llm-d URL
* semantic cache settings
* RAG settings
* asset paths
* tokenizer path
* feature flags

### `deploy/k8s-fastapi/fastapi-secret.fullstack.yaml`

Main secret config for FastAPI.

Contains things like:

* Milvus credentials
* Seaweed/S3 credentials

### `deploy/k8s-fastapi/fastapi-deployment.fullstack.yaml`

Main FastAPI deployment.

Mounts:

* `rag-store-tenants-pvc`

Uses:

* `llm-service-kernel:fastapi-selfcontained`

### `deploy/k8s-fastapi/fastapi-service.yaml`

Exposes FastAPI as a Kubernetes Service.

### `deploy/k8s-fastapi/fastapi-bootstrap-job.yaml`

Runs one bootstrap job that ensures:

* semantic cache Mongo indexes exist
* semantic cache Milvus collection exists
* Seaweed bucket exists
* seeded tenant manifest directories are copied to the PVC if missing

### `deploy/k8s-fastapi/rag-store-tenants-pvc.yaml`

Persistent storage for `/rag_store_tenants`.

### `deploy/k8s-fastapi/tenant-ingest-input-pvc.yaml`

Persistent storage used to stage tenant PDFs before ingestion.

---

## Service code files

### `src/service/config.py`

Main runtime configuration loader.

Reads environment variables and exposes settings like:

* backend selection
* semantic cache configuration
* RAG configuration
* object store configuration
* tokenizer/model paths

### `src/service/bootstrap.py`

Builds the runtime used by FastAPI.

Creates:

* model backend client
* semantic cache client
* RAG router
* tokenizer
* orchestrator

### `src/service/bootstrap_init.py`

Bootstrap job entrypoint.

This is not the serving path.
This is the **cluster initialization path**.

### `src/service/main.py`

FastAPI app entrypoint.

### `src/service/cache/semantic_gptcache.py`

Semantic cache implementation using:

* Mongo
* Milvus
* local embedding model

### `src/service/rag/tenant_router.py`

Routes RAG retrieval by tenant.

### `src/service/rag/milvus_rag.py`

Milvus-backed retriever for tenant chunks.

### `src/service/rag/seaweed_chunk_store.py`

Loads chunk payloads from SeaweedFS object storage.

### `src/service/clients/vllm_client.py`

Wrapper for generation backend access.

Currently supports:

* direct vLLM
* llm-d

### `src/service/orchestrator/chat.py`

Main orchestration logic for:

* semantic cache lookup
* RAG retrieval
* backend call
* response handling

### `src/service/observability/viz.py`

Additional FastAPI visualization/debug routes.

This is not a separate inference architecture path.
It is a browser-accessible helper layer for visualization/inspection/debugging on top of the same service.

---

## Ingestion files

### `scripts/start_tenant_ingest_uploader.sh`

Creates a simple uploader pod that mounts the tenant ingest PVC.

Used as a target for `kubectl cp`.

### `scripts/upload_tenant_pdfs.sh`

Copies local PDFs into the ingest PVC for a tenant.

### `scripts/run_tenant_ingest_job.sh`

Launches a Kubernetes Job to ingest one tenant.

### `scripts/ingest_tenant_to_milvus.py`

Actual ingestion logic.

It:

* reads PDFs
* extracts text using `pypdf`
* chunks text
* embeds chunks using BGE
* uploads PDFs to Seaweed
* uploads chunk JSON payloads to Seaweed
* inserts vectors into Milvus
* writes tenant manifest file

### `scripts/chat_cli.py`

Main CLI for manual testing of the service.

Supports:

* base URL
* tenant selection
* debug output

---

# 3. Preconditions

Before running the official flow, make sure:

* Minikube is running
* the `llm-d` side is already available in-cluster
* you are inside:

```bash
cd /home/mohamad/LLM-end-to-end-Service-main/llm-service-kernel
```

Useful checks:

```bash
minikube status
kubectl get nodes
kubectl get ns
```

---

# 4. Official build and deploy flow

## Step 1: prepare runtime assets

```bash
bash scripts/prepare_fastapi_runtime_assets.sh
```

What this does:

* creates `fastapi_runtime_assets/`
* copies embedding models
* copies tokenizer assets
* copies seed tenant manifests

Check:

```bash
find fastapi_runtime_assets -maxdepth 3 | sort
```

You should see:

* `fastapi_runtime_assets/models/...`
* `fastapi_runtime_assets/rag_store_tenants/...`

---

## Step 2: vendor offline PDF parser

```bash
bash scripts/vendor_local_pypdf.sh
```

What this does:

* copies locally installed `pypdf` into `vendor/`

Check:

```bash
find vendor -maxdepth 2 | sort
```

You should see:

* `vendor/pypdf`
* `vendor/pypdf-<version>.dist-info`

---

## Step 3: point Docker to Minikube

```bash
eval "$(minikube -p minikube docker-env)"
```

Why:

* ensures the built image is available to Kubernetes without pushing to a remote registry

Check:

```bash
docker images | grep llm-service-kernel
```

---

## Step 4: build the self-contained image

```bash
docker build -f Dockerfile.service -t llm-service-kernel:fastapi-selfcontained .
```

What this image contains:

* app code
* scripts
* runtime assets
* vendored `pypdf`

---

# 4A. Generation backend configuration

The current architecture does not change when switching backend or served model.

The same service can target different generation backends through config.

## Canonical generation variables

The main variables are:

- `GENERATION_BACKEND`
- `GENERATION_BASE_URL`
- `GENERATION_API_MODE`
- `GENERATION_MODEL_NAME`

### Example of current config:

GENERATION_BACKEND=llmd
GENERATION_BASE_URL=http://infra-local-inference-gateway-istio.llm-d-local.svc.cluster.local:80
GENERATION_API_MODE=completions
GENERATION_MODEL_NAME=qwen2.5-0.5b


These are set in:

```bash
deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml
```
## Step 5: deploy fullstack

```bash
bash scripts/deploy_fastapi_fullstack.sh
```

What this script does:

1. applies Mongo, Milvus, Seaweed manifests
2. applies PVCs for tenant manifests and ingest input
3. applies FastAPI ConfigMap and Secret
4. runs bootstrap job
5. deploys/restarts FastAPI
6. waits for rollout

---

# 5. What to check after deploy

## Check pods

```bash
kubectl get pods -n llm-service
```

Expected important pods:

* `mongodb-...`
* `milvus-...`
* `milvus-etcd-...`
* `milvus-minio-...`
* `seaweed-master-...`
* `seaweed-volume-...`
* `seaweed-filer-...`
* `seaweed-s3-...`
* `llm-service-kernel-...`
* `llm-service-kernel-bootstrap-...` should be `Completed`

---

## Check bootstrap job logs

```bash
kubectl logs -n llm-service job/llm-service-kernel-bootstrap
```

Expected output should include:

* mongo semantic-cache indexes ok
* milvus semantic-cache collection ok
* seaweed bucket ok
* seed rag manifests ok
* bootstrap complete

---

## Check deployment rollout

```bash
kubectl rollout status deployment/llm-service-kernel -n llm-service --timeout=300s
```

---

## Check service health

Port-forward in one terminal:

```bash
kubectl port-forward -n llm-service svc/llm-service-kernel 8080:8080
```

Then in another terminal:

```bash
curl http://127.0.0.1:8080/health
```

Important fields to verify:

* `"ok": true`
* `"model_backend": "llmd"` or expected backend
* `"semantic_cache_runtime_enabled": true`
* `"rag_runtime_enabled": true`

---

# 6. How to ingest PDFs for a tenant

## Step 1: start uploader pod

```bash
bash scripts/start_tenant_ingest_uploader.sh
```

Check:

```bash
kubectl get pods -n llm-service | grep tenant-ingest-uploader
```

Expected:

* uploader pod is `Running`

---

## Step 2: upload local PDFs

Example:

```bash
bash scripts/upload_tenant_pdfs.sh tenantC /path/to/local/pdfs
```

What this does:

* creates `/tenant_ingest_input/tenantC` inside the uploader pod
* copies your local PDFs there

Check what is inside the ingest PVC:

```bash
kubectl exec -n llm-service tenant-ingest-uploader -- sh -c 'find /tenant_ingest_input -maxdepth 3 -type f | sort'
```

---

## Step 3: run tenant ingest job

```bash
bash scripts/run_tenant_ingest_job.sh tenantC
```

What this does:

* creates a Kubernetes Job
* reads PDFs from `/tenant_ingest_input/tenantC`
* extracts text
* chunks and embeds text
* uploads PDFs and chunk objects to Seaweed
* inserts vectors into Milvus
* writes `/rag_store_tenants/tenantC/manifest.json`

---

## Step 4: check ingest result

See jobs:

```bash
kubectl get jobs -n llm-service | grep tenant-ingest
```

See pods:

```bash
kubectl get pods -n llm-service | grep tenant-ingest
```

Read latest job logs:

```bash
kubectl logs -n llm-service job/$(kubectl get jobs -n llm-service --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
```

Expected output should include lines like:

* `pdfs=<N>`
* `extracted chunks=<N>`
* `uploaded pdf objects=<N>`
* `uploaded chunk objects=<N>`
* `inserted rows=<N>`
* `wrote manifest: /rag_store_tenants/<tenant>/manifest.json`

---

## Step 5: check manifest visibility from FastAPI pod

```bash
kubectl exec -n llm-service deploy/llm-service-kernel -- sh -c 'find /rag_store_tenants -maxdepth 3 | sort'
```

You should see the tenant directory and `manifest.json`.

---

# 7. How to test the service

## Start port-forward

```bash
kubectl port-forward -n llm-service svc/llm-service-kernel 8080:8080
```

Keep this terminal open.

---

## Open CLI against a tenant

Example:

```bash
python scripts/chat_cli.py --base-url http://127.0.0.1:8080 --tenant tenantC --show-debug
```

What to test:

* ask one question clearly answered by the uploaded PDFs
* ask it again a second time

What to expect:

* first time: likely `rag_plus_backend`
* second time: possibly `semantic_cache`

Important debug fields:

* `tenant: tenantC`
* `rag_retrieved: True`
* `rag_used: True`
* `sources: [...]`
* later repeated query may show `route_taken: semantic_cache`

---

# 8. Useful Kubernetes commands

## General inspection

```bash
kubectl get ns
kubectl get pods -A
kubectl get svc -A
kubectl get pvc -A
kubectl get jobs -A
```

## Namespace-specific inspection

```bash
kubectl get pods -n llm-service
kubectl get svc -n llm-service
kubectl get pvc -n llm-service
kubectl get jobs -n llm-service
```

## Watch pods live

```bash
kubectl get pods -n llm-service -w
```

## Describe a failing pod

```bash
kubectl describe pod -n llm-service <pod-name>
```

## Read pod logs

```bash
kubectl logs -n llm-service <pod-name>
```

## Read job logs

```bash
kubectl logs -n llm-service job/<job-name>
```

## Exec into FastAPI pod

```bash
kubectl exec -it -n llm-service deploy/llm-service-kernel -- sh
```

## Exec into uploader pod

```bash
kubectl exec -it -n llm-service tenant-ingest-uploader -- sh
```

## Restart FastAPI deployment

```bash
kubectl rollout restart deployment/llm-service-kernel -n llm-service
kubectl rollout status deployment/llm-service-kernel -n llm-service --timeout=300s
```

## Delete old ingest jobs

```bash
kubectl delete job -n llm-service $(kubectl get jobs -n llm-service -o name | grep tenant-ingest | sed 's#job.batch/##')
```

## Delete uploader pod

```bash
bash scripts/stop_tenant_ingest_uploader.sh
```

---

# 9. Common checks and failure cases

## Case: bootstrap job fails

Check:

```bash
kubectl logs -n llm-service job/llm-service-kernel-bootstrap
kubectl describe job -n llm-service llm-service-kernel-bootstrap
```

Possible causes:

* Milvus not ready yet
* Mongo not ready
* Seaweed not ready
* secret/config mismatch

---

## Case: FastAPI health fails

Check:

```bash
kubectl get pods -n llm-service
kubectl logs -n llm-service deploy/llm-service-kernel
curl http://127.0.0.1:8080/health
```

Possible causes:

* backend URL wrong
* tokenizer/model path wrong
* bootstrap not completed
* secret/config mismatch

---

## Case: tenant ingest job fails

Check:

```bash
kubectl get jobs -n llm-service | grep tenant-ingest
kubectl get pods -n llm-service | grep tenant-ingest
kubectl logs -n llm-service job/<job-name>
kubectl describe pod -n llm-service <pod-name>
```

Also verify PDFs exist:

```bash
kubectl exec -n llm-service tenant-ingest-uploader -- sh -c 'find /tenant_ingest_input -maxdepth 3 -type f | sort'
```

Possible causes:

* missing PDFs in input directory
* missing BGE path
* object store secret mismatch
* schema mismatch in Milvus
* missing vendored `pypdf`

---

## Case: RAG not used

Check the CLI debug block.

Possible reasons:

* `rag_retrieved: True` but `rag_used: False`
* top score below threshold
* tenant mismatch
* question not actually answered by the uploaded PDFs

You can inspect:

* `top_score`
* `score_threshold`
* listed sources

---

## Case: semantic cache not hitting

This is normal for the first request.

Ask the same question again.

Check debug fields:

* `semantic_enabled`
* `semantic_hit`
* `semantic_reject_reason`

---

# 10. Canonical usage summary

## Build and deploy

```bash
bash scripts/prepare_fastapi_runtime_assets.sh
bash scripts/vendor_local_pypdf.sh
eval "$(minikube -p minikube docker-env)"
docker build -f Dockerfile.service -t llm-service-kernel:fastapi-selfcontained .
bash scripts/deploy_fastapi_fullstack.sh
```

## Upload and ingest PDFs

```bash
bash scripts/start_tenant_ingest_uploader.sh
bash scripts/upload_tenant_pdfs.sh tenantC /path/to/local/pdfs
bash scripts/run_tenant_ingest_job.sh tenantC
```

## Test chat

```bash
kubectl port-forward -n llm-service svc/llm-service-kernel 8080:8080
```
Then check health or just run without port forwarding:
```bash

python scripts/chat_cli.py --base-url http://127.0.0.1:8080 --tenant tenantC --show-debug
```

---

# 11. Current design notes

## What is already cleaned up

* no FastAPI dependency on `minikube mount`
* runtime assets are packaged into the image
* bootstrap is automatic
* tenant ingest runs in-cluster
* `llama_index` dependency was removed from ingestion
* offline `pypdf` is vendored into the image


# 12. Minimal commands for day-to-day use

## Fresh build/deploy

```bash
bash scripts/prepare_fastapi_runtime_assets.sh
bash scripts/vendor_local_pypdf.sh
eval "$(minikube -p minikube docker-env)"
docker build -f Dockerfile.service -t llm-service-kernel:fastapi-selfcontained .
bash scripts/deploy_fastapi_fullstack.sh
```

## New tenant ingest

```bash
bash scripts/start_tenant_ingest_uploader.sh
bash scripts/upload_tenant_pdfs.sh tenantC /path/to/local/pdfs
bash scripts/run_tenant_ingest_job.sh tenantC
```

## Test tenant

```bash
kubectl port-forward -n llm-service svc/llm-service-kernel 8080:8080

curl http://127.0.0.1:8080/health
```
For this, you do not need the port forwarding active since chat_cli.py in /scripts handles the startup automatically. 
```bash
python scripts/chat_cli.py --base-url http://127.0.0.1:8080 --tenant tenantC --show-debug
```
```
```
```
