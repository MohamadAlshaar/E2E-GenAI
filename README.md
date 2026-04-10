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
