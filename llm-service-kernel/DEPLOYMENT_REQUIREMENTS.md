# Deployment Requirements

## Purpose

This document defines what is required to deploy the current **single-node fullstack** version of the service.

It is written for the current architecture and current implementation, without changing the system design.

The goal is to make it clear:

- what infrastructure is needed
- what software must already exist
- what is packaged in this repo
- what is still expected from the environment
- what is required for local/offline deployment
- what must be ready before moving to a real server

---

# 1. Current deployment target

## Current supported target

The current deployment target is:

- **one Kubernetes node**
- **single-node fullstack**
- full path working:
  - FastAPI
  - semantic cache
  - RAG
  - object storage
  - vector database
  - generation backend path
- intended current environment:
  - **Minikube**
  - one machine
  - reproducible local deployment

## Current architecture

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
