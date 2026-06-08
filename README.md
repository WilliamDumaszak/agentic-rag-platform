# Agentic RAG Platform

Agentic Retrieval-Augmented Generation platform using LangGraph, with adaptive routing, hybrid retrieval support, response ranking, and evaluation endpoints.

## Overview

This project builds a graph-based RAG agent that decides the best path per query:
- Retrieve from local/cloud knowledge base
- Search the web when needed
- Generate direct response when retrieval is unnecessary
- Rank candidate answers by semantic similarity

It also includes:
- Document ingestion
- RAG quality evaluation (classic metrics + RAGAS)
- FastAPI interface for query/ingestion/evaluation
- Docker-based local runtime

## Architecture

```text
User query
  -> decision node (route)
    -> retrieve (Chroma or Azure AI Search)
    -> web search fallback
    -> direct generation
  -> generate N candidates
  -> rank candidates
  -> return best answer + confidence + source
```

## Repository Structure

```text
agentic-rag-platform/
  config/
    config.yaml
  agent/
    state.py
    nodes.py
    graph.py
  rag/
    ingestion.py
    azure_search.py
  evaluation/
    metrics.py
    ragas_eval.py
  api/
    main.py
    schemas.py
  tests/
    test_agent.py
    test_api.py
  docker-compose.yml
  Dockerfile
  requirements.txt
  .env.example
```

## Local Run

### 1) Setup

```bash
cd agentic-rag-platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start dependencies (Ollama)

```bash
docker compose --profile setup up ollama-pull
```

### 3) Ingest documents

```bash
curl -X POST http://localhost:8000/ingest
```

### 4) Query the agent

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is supply chain management?"}'
```

### 5) Run RAGAS evaluation

```bash
curl -X POST http://localhost:8000/evaluate/ragas \
  -H "Content-Type: application/json" \
  -d '[{"question":"What is logistics?","answer":"Logistics is...","contexts":["Logistics refers to..."]}]'
```

## Configuration Highlights

See `config/config.yaml` for:
- LLM provider switching (`ollama`, `openai`, `azure_openai`)
- Vector DB provider switching (`chroma`, `azure_search`)
- Hybrid retrieval settings (`retrieval.provider`, `rrf_k`, `top_k`)
- Human-in-the-loop queue (`hitl.db_path`)
- Prompt versioning (`prompts.active_version`)
- Document Intelligence provider (`local` or `azure`)

## CI/CD

Workflow: `.github/workflows/pipeline.yml`
- Test job on push/PR
- Build and push images on `main`
- Supports Docker Hub and Azure Container Registry

## Notes

- For Azure OpenAI or Azure Search, define environment variables from `.env.example`.
- Keep local provider defaults for quick local development.

## License

This project follows the repository root license.
