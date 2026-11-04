# DeepSearch-Local

> **Autonomous Deep Research Agent Platform**  
> Running locally on consumer GPUs (NVIDIA RTX 3050 6GB VRAM) with zero cloud API costs.

---

## Overview

**DeepSearch-Local** is an open-source, local-first autonomous deep research platform designed to rival commercial systems such as *Perplexity Pro Search* and *OpenAI Deep Research*.

It pairs a parameter-efficient fine-tuned (PEFT / QLoRA) **Small Language Model (SLM, 1.5B–3B parameters)** with a cyclic multi-step **LangGraph** agent, Anthropic's **Model Context Protocol (MCP)** tool execution, a dynamic **Qdrant Hybrid RAG** engine, and a **Next.js** interactive web dashboard.

---

## Architecture

```
User Prompt (Next.js Dashboard)
       │ (SSE / REST)
       ▼
FastAPI Orchestrator
       │
       ▼
LangGraph Agentic State Machine
 ├── 1. Planner Node
 ├── 2. Search & Scrape Node (MCP Server: DuckDuckGo + Trafilatura)
 ├── 3. Dynamic Ingestion Node (Qdrant + BM25)
 ├── 4. Grounded Synthesis Node (Hybrid RAG + BGE Reranker)
 ├── 5. Gap & Reflection Node (Loops back if gaps found, max 2 iterations)
 └── 6. Report Compiler Node (Deduplicated citations & final markdown)
```

---

## Repository Structure

```
deepsearch-local/
├── docker-compose.yml              # Qdrant and Langfuse local stack
├── pyproject.toml                  # Python dependencies managed via uv
├── .env.example                    # Environment variable template
│
├── agent/                          # LangGraph state machine & prompts
├── mcp_server/                     # Model Context Protocol JSON-RPC tools
├── rag/                            # Dynamic Hybrid RAG (Qdrant + BM25 + Reranking)
├── finetuning/                     # 4-bit QLoRA pipeline & synthetic data generator
├── api/                            # FastAPI backend with SSE streaming
├── eval/                           # Ragas & TruLens comparative benchmark suite
├── web/                            # Next.js 14+ frontend dashboard
└── tests/                          # Test suite
```

---

## Quickstart

### 1. Prerequisites
- Linux OS with NVIDIA GPU (e.g. RTX 3050 6GB) and CUDA 12.x/13.x
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Docker & Docker Compose](https://docs.docker.com/compose/)
- [Node.js](https://nodejs.org/) (v20+) & npm

### 2. Environment Setup
```bash
cp .env.example .env
```

### 3. Start Local Infrastructure
```bash
docker compose up -d
```

### 4. Install Dependencies
```bash
# Core backend dependencies
uv sync

# Optional: Finetuning dependencies (PyTorch + CUDA)
uv sync --group finetuning
```

---

## License
MIT
