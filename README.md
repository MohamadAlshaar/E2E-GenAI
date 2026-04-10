# E2E GenAI Service — LLM Service Kernel

Single-node Kubernetes deployment of a production-style GenAI inference stack:
**RAG retrieval** (Milvus + SeaweedFS) → **Semantic caching** (Milvus + MongoDB) → **LLM inference** (vLLM via llm-d).

## Quick Start

```bash
# 1. One-time machine setup (installs deps, downloads models, starts minikube)
./setup.sh

# 2. Deploy everything + ingest RAG documents
./deploy.sh

# 3. Chat
python3 scripts/chat_cli.py --show-debug
```

Total time: ~15 min for setup, ~15 min for deploy (mostly GPU model loading).

## Prerequisites

- Ubuntu 22.04+ with **NVIDIA GPU** and driver installed (`nvidia-smi` must work)
- 16GB+ RAM, 8+ CPU cores, 80GB+ disk
- Internet access (for model download on first setup)

`setup.sh` handles everything else: Docker, NVIDIA Container Toolkit, minikube, kubectl, Helm, Istio, ML models, sample documents.

## Architecture

```
Client → FastAPI Orchestrator → Semantic Cache (MiniLM + Milvus + MongoDB)
                               → RAG Retrieval (BGE + Milvus + SeaweedFS)
                               → LLM Backend (vLLM via llm-d gateway)
```

All components run on a single minikube node across 3 namespaces:
- `istio-system` — Istio service mesh + Gateway API
- `llm-d-local` — vLLM inference pods + llm-d routing
- `llm-service` — FastAPI, Milvus, MongoDB, SeaweedFS

## Changing the LLM Model

1. Download the new model to the repo root (e.g., `Llama-3.2-1B-Instruct/`)
2. Edit `scripts/provision_model_artifacts.sh` — set `MODEL_SOURCE_DIR`
3. Edit `deploy/llmd-local/modelservice-values.yaml`:
   - `modelArtifacts.name` — the served model name
   - `decode.containers[0].args` — the `--served-model-name` and `--max-model-len` flags
   - `decode.containers[0].resources` — GPU/memory limits
4. Edit `deploy/k8s-fastapi/fastapi-configmap.fullstack.yaml` — set `GENERATION_MODEL_NAME`
5. Re-run `./deploy.sh`

## Adding More vLLM Workers

Edit `deploy/llmd-local/modelservice-values.yaml`:
```yaml
decode:
  replicas: 2        # scale out decode workers
  parallelism:
    tensor: 1         # tensor parallelism per worker
```

The llm-d gateway routes requests across all workers automatically.

## RAG: Adding Your Own Documents

Place PDF files in `docs_RAG/` and re-run deployment:
```bash
FORCE_REINGEST=1 ./deploy.sh
```

Or ingest a different tenant:
```bash
TENANT_ID=myTenant FORCE_REINGEST=1 ./deploy.sh
```

## Environment Variables

### setup.sh
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_HOST_BOOTSTRAP` | `0` | Skip system dep installation |
| `SKIP_MODEL_DOWNLOAD` | `0` | Skip HuggingFace model download |
| `SKIP_DOCS_DOWNLOAD` | `0` | Skip sample PDF download |
| `MINIKUBE_CPUS` | `8` | CPU cores for minikube |
| `MINIKUBE_MEMORY` | `16384` | RAM in MB for minikube |

### deploy.sh
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_DEPLOY` | `0` | Skip K8s deployment |
| `SKIP_INGEST` | `0` | Skip RAG ingestion |
| `FORCE_REINGEST` | `0` | Drop and re-ingest RAG collection |
| `TENANT_ID` | `tenantA` | Tenant for RAG ingestion |
| `FASTAPI_LOCAL_PORT` | `18081` | Local port for FastAPI |

## How Everything Is Packaged and Deployed

Nothing is installed on the host machine directly (except NVIDIA drivers and minikube). Every service runs as a **Docker container** pulled from a public registry, orchestrated by **Kubernetes** (minikube). The repo contains only declarative manifests and Helm charts — Kubernetes reads them, pulls the required images, and starts containers from them automatically.

### Service Images

| Component | Docker Image | Deployed Via | What It Does |
|---|---|---|---|
| **Milvus** (vector DB) | `milvusdb/milvus:v2.6.11` | `deploy/k8s-storage/milvus.yaml` | Stores RAG chunks and semantic-cache embeddings as vector collections |
| **etcd** (Milvus metadata) | `quay.io/coreos/etcd:v3.5.18` | same file | Key-value store that Milvus uses for internal metadata and coordination |
| **MinIO** (Milvus object store) | `minio/minio:RELEASE.2023-03-20T...` | same file | S3-compatible blob store that Milvus uses for segment storage |
| **MongoDB** (cache payloads) | `mongo:8` | `deploy/k8s-storage/mongo.yaml` | Stores full cached LLM responses for the semantic cache layer |
| **SeaweedFS** (RAG chunk store) | `chrislusf/seaweedfs:latest` | `deploy/k8s-storage/seaweedfs.yaml` | S3-compatible object store holding raw PDF text chunks for RAG retrieval (4 pods: master, volume, filer, s3) |
| **vLLM** (LLM inference) | `ghcr.io/llm-d/llm-d-cuda:v0.6.0` | Helm chart `llm-d-modelservice-v0.4.8.tgz` | GPU-accelerated inference server running the Qwen model, OpenAI-compatible API |
| **llm-d gateway** (routing) | `ghcr.io/llm-d/llm-d-routing-sidecar:v0.7.1` | Helm chart `llm-d-infra-v1.3.10.tgz` | Routes inference requests across vLLM workers with load balancing |
| **FastAPI orchestrator** | Built locally from `src/service/` | `deploy/k8s-fastapi/` | The main application — routes queries through semantic cache → RAG → LLM |
| **Istio** (service mesh) | Public Istio images | Installed by `setup.sh` via `istioctl` | Provides the Gateway API that fronts the llm-d inference endpoint |

When `deploy.sh` runs `kubectl apply -f deploy/k8s-storage/milvus.yaml`, Kubernetes reads the manifest, sees it needs the `milvusdb/milvus:v2.6.11` image, pulls it from Docker Hub, and starts a container from it as a Pod. The same happens for every service — manifests declare what to run, Kubernetes handles the container lifecycle. Helm charts (`.tgz` files) work the same way but with templating — `helm install` renders the templates into Kubernetes manifests and applies them.

### ML Models

The three ML models are **not baked into Docker images**. They are downloaded from HuggingFace by `scripts/download_models.sh` and then provided to the containers that need them:

| Model | Purpose | How It Reaches the Container |
|---|---|---|
| **Qwen2.5-0.5B-Instruct** | LLM generation | Copied into minikube's node filesystem at `/data/qwen-model` via `scripts/provision_model_artifacts.sh`, then mounted into the vLLM pod as a PersistentVolume |
| **bge-base-en-v1.5** | RAG embedding (768-dim) | Bundled into the FastAPI Docker image at build time via `scripts/prepare_fastapi_runtime_assets.sh` |
| **all-MiniLM-L6-v2** | Semantic cache embedding (384-dim) | Same — bundled into the FastAPI Docker image at build time |

### Helm Charts

The two `.tgz` Helm chart archives in `deploy/llmd-local/` are **pre-packaged and committed to the repo**. They are not downloaded at deploy time. `helm install` reads them directly, renders Kubernetes manifests from the templates inside, and applies them to the cluster.

### What Happens on a Fresh Machine

`setup.sh` prepares the machine:
1. Installs NVIDIA GPU drivers (if missing)
2. Installs system dependencies: Docker, NVIDIA Container Toolkit, kubectl, Helm, minikube, Istio
3. Starts minikube with GPU passthrough and deploys the NVIDIA device plugin
4. Downloads the 3 ML models from HuggingFace
5. Downloads sample PDF documents from arxiv

`deploy.sh` deploys the stack:
1. `kubectl apply` the YAML manifests → Kubernetes pulls Docker images from public registries and starts containers
2. `helm install` the llm-d charts → same process for the inference gateway and vLLM
3. Builds the FastAPI Docker image locally inside minikube's Docker daemon
4. Runs the RAG ingestion pipeline (PDF → embeddings → Milvus + SeaweedFS)
5. Verifies inference with a smoke test
6. Sets up port-forwards so you can reach the services from `localhost`

## Project Layout

```
llm-service-kernel/
  setup.sh                    # One-time machine setup
  deploy.sh                   # Full stack deploy + ingest
  scripts/
    deploy_fullstack_single_node.sh   # K8s deployment orchestrator
    bootstrap_host_ubuntu.sh          # System dependency installer
    download_models.sh                # HuggingFace model downloader
    download_sample_docs.sh           # Sample PDF downloader
    ingest_tenant_to_milvus.py        # RAG ingestion pipeline
    chat_cli.py                       # Interactive test CLI
    provision_model_artifacts.sh      # Copy LLM model into minikube
    prepare_fastapi_runtime_assets.sh # Bundle models for Docker image
  src/service/                # FastAPI application source
  deploy/
    k8s-storage/              # Milvus, MongoDB, SeaweedFS manifests
    k8s-fastapi/              # FastAPI ConfigMap, Secrets, Deployment
    llmd-local/               # llm-d Helm charts + values
```
