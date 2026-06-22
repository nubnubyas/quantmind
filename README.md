# QuantMind — Quant Research & Career AI Copilot

> An AI copilot for quant researchers and AI engineer job seekers: search research papers, explain quant concepts, generate backtesting code, plan research roadmaps, remember user preferences, and assist with quant/AI job applications.
>
> **Status**: Phase 2 complete (SP3 achieved) · Phase 3 starting

---

## What is QuantMind?

QuantMind is a multi-subgraph LangGraph agent that combines **hybrid RAG** (sparse + dense vector search), **code generation + sandbox verification**, and **job application tracking** into a single copilot. It targets two personas:

- **Quant researchers** who need to search papers, understand concepts, and prototype backtesting code.
- **AI/quant job seekers** who need interview prep, application tracking, and career planning.

### User Scenarios (10 scenarios, 120-question benchmark)

| # | Scenario | Description |
|---|----------|-------------|
| S1 | Paper Search | Search for quant research papers by topic, author, or methodology |
| S2 | Concept Explanation | Explain quant concepts (e.g., Black-Scholes, Kalman filters, factor models) |
| S3 | Code Generation | Generate backtesting / data pipeline code with sandbox execution and HITL review |
| S4 | Research Planning | Break a research idea into a phased plan with milestones |
| S5 | Interview Prep | Generate technical quant/AI interview questions with answers |
| S6 | Job Tracking | Log applications, track status, and manage follow-ups |
| S7 | Paper Summarization | Summarize key findings from uploaded or searched papers |
| S8 | User Memory | Remember user preferences, background, and past interactions across sessions |
| S9 | Market Data | Fetch and analyze market data (price, fundamentals, options) |
| S10 | Resume Feedback | Provide feedback on quant resumes and cover letters |

---

## Features

### Core Capabilities

- **Hybrid RAG Pipeline** — Sparse (BM25 via Qdrant) + Dense (fastembed `BAAI/bge-small-en-v1.5`) vector search with reciprocal rank fusion for paper retrieval.
- **CodeGen + Sandbox** — LLM generates backtesting code, which runs in an isolated Docker sandbox. Human-in-the-loop interrupt gate lets users review code before execution.
- **Multi-Subgraph Architecture** — Research subgraph and CodeGen subgraph are composed in a parent LangGraph `StateGraph`, each with its own tool set and routing logic.
- **Three-Tier Persistence** — LangGraph Checkpointer (SQLite) for conversation state, LangGraph Store for cross-conversation memory, and PostgreSQL for structured data (job applications, notes, ingestion logs).
- **Human-in-the-Loop (HITL)** — `interrupt()` before sandbox code execution; user can approve, reject, or modify generated code via the FastAPI resume endpoint.
- **LLMOps with LangSmith** — Full tracing of every tool call, LLM invocation, and subgraph transition. Supports offline evaluation against the benchmark dataset.
- **8 Tools (T1–T8)** — `search_papers`, `fetch_paper_details`, `explain_concept`, `generate_backtest_code`, `create_research_plan`, `generate_interview_questions`, `track_application`, `update_user_memory`.
- **Long-Term Memory** — `UserMemoryStore` persists user profiles, research interests, and interaction history across sessions using LangGraph's `Store` API.

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Agent Framework | LangGraph 1.2+ | Multi-subgraph orchestration, state management, HITL interrupts |
| LLM (fast) | DeepSeek V3 (`deepseek-chat`) | Tool calling, paper summarization, code generation |
| LLM (strong) | DeepSeek R1 (`deepseek-reasoner`) | Complex reasoning, research planning |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5`) | Dense vector embeddings (384-dim) |
| Vector DB | Qdrant 1.18 | Hybrid sparse + dense search with RRF fusion |
| API Server | FastAPI | `/chat`, `/interrupt`, `/resume` endpoints |
| Sandbox | Docker (subprocess) | Isolated code execution for generated backtesting scripts |
| Structured DB | PostgreSQL | Job applications, notes, ingestion logs |
| Tracing | LangSmith | LLMOps tracing, evaluation, and debugging |
| Data Ingestion | `arxiv` + PyMuPDF + tiktoken | Paper fetching, PDF parsing, and token-aware chunking |

---

## Architecture

### Subgraph Architecture

```text
                    ┌──────────────────────────┐
                    │     Parent Graph          │
                    │  (router + state merge)   │
                    └──────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
   ┌───────────▼───────────┐   ┌───────────────▼──────────┐
   │   Research Subgraph   │   │    CodeGen Subgraph       │
   │                       │   │                           │
   │  T1: search_papers    │   │  T4: generate_backtest_code│
   │  T2: fetch_details    │   │  Sandbox: execute + verify  │
   │  T3: explain_concept  │   │  HITL: interrupt() gate    │
   │  T5: create_plan      │   │  Resume: approve/modify    │
   │  T6: interview_qs     │   │                           │
   │  T7: track_application│   │                           │
   │  T8: update_memory    │   │                           │
   └───────────────────────┘   └───────────────────────────┘
```

### Hybrid RAG Pipeline

```text
User Query
    │
    ├──► Sparse Embedding (BM25 via Qdrant)
    │
    ├──► Dense Embedding (fastembed BAAI/bge-small-en-v1.5)
    │
    └──► Qdrant Hybrid Search (Reciprocal Rank Fusion)
            │
            ▼
       Ranked Paper Chunks + Metadata
            │
            ▼
       LLM Response (with citations)
```

### Three-Tier Persistence

| Tier | Backend | Content | Lifecycle |
|------|---------|---------|-----------|
| Checkpointer | SQLite (`langgraph-checkpoint-sqlite`) | Per-conversation graph state | Per session |
| Store | LangGraph `Store` (SQLite-backed) | Cross-conversation user memory | Persistent |
| Structured | PostgreSQL | Job applications, notes, ingestion logs | Persistent |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant, PostgreSQL, and sandbox)
- DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com))
- LangSmith API key (optional, for tracing — [smith.langchain.com](https://smith.langchain.com))

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/quantmind.git
cd quantmind

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # dev tools + testing

# 4. Configure environment
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY and (optionally) LANGSMITH_API_KEY

# 5. Start infrastructure (Qdrant + PostgreSQL via Docker)
# Run Qdrant:
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
# Run PostgreSQL:
docker run -d --name quantmind-postgres \
  -e POSTGRES_USER=quantmind -e POSTGRES_PASSWORD=quantmind \
  -e POSTGRES_DB=quantmind -p 5432:5432 postgres:16

# 6. Ingest sample papers (optional)
python -m src.ingestion.pipeline --source data/sample_papers/

# 7. Run the demo
python scripts/run_research_demo.py
```

### Run the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `POST /chat` — Send a message, receive streaming or complete response
- `GET /interrupts` — List pending HITL interrupts
- `POST /resume` — Resume a paused graph execution

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_tools.py -v
```

---

## Project Structure

```text
quant-project/
├── README.md
├── .env.example                    # Environment variable template
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev dependencies
│
├── src/
│   ├── agents/
│   │   ├── research_agent.py       # Research subgraph (T1-T3, T5-T8)
│   │   ├── codegen_agent.py        # CodeGen subgraph (T4 + sandbox + HITL)
│   │   └── state.py                # Shared TypedDict state
│   ├── tools/
│   │   ├── search_papers.py        # T1: Hybrid RAG paper search
│   │   ├── fetch_paper_details.py  # T2: arXiv metadata + PDF
│   │   ├── explain_concept.py      # T3: Concept explanation
│   │   ├── generate_backtest_code.py # T4: Backtesting code generation
│   │   ├── create_research_plan.py # T5: Research roadmapping
│   │   ├── generate_interview_questions.py # T6: Interview Q&A
│   │   ├── track_application.py    # T7: Job application CRUD
│   │   ├── update_user_memory.py   # T8: Long-term memory persistence
│   │   ├── _helpers.py             # Shared tool utilities
│   │   └── types.py                # Tool Pydantic models
│   ├── vector_store/
│   │   ├── qdrant_client_wrapper.py # Qdrant sparse + dense search
│   │   ├── mock_vector_store.py     # In-memory mock for testing
│   │   └── types.py                 # Vector store types
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py        # ArXiv API + PDF download
│   │   ├── pdf_parser.py           # PyMuPDF extraction
│   │   ├── chunker.py              # Token-aware text chunking
│   │   └── pipeline.py             # End-to-end ingestion pipeline
│   ├── sandbox/
│   │   └── sandbox_runner.py       # Docker sandbox for code execution
│   ├── memory/
│   │   └── user_memory.py          # LangGraph Store-based user memory
│   ├── config/
│   │   └── llm_client.py           # DeepSeek client (OpenAI-compatible)
│   ├── db/
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── base.py                  # Session + engine
│   │   ├── application_repo.py     # Job application repository
│   │   ├── notes_repo.py           # Notes repository
│   │   ├── ingestion_log_repo.py   # Ingestion log repository
│   │   └── exceptions.py           # Custom DB exceptions
│   ├── api/
│   │   ├── main.py                 # FastAPI app + routes
│   │   └── models.py               # Request/response Pydantic models
│   ├── eval/                        # Evaluation framework (placeholder)
│   └── ui/                          # Streamlit UI (placeholder)
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (LLM mock, Qdrant mock, etc.)
│   └── unit/
│       ├── test_tools.py            # Tool execution tests (T1-T8)
│       ├── test_codegen_agent.py    # CodeGen subgraph + HITL tests
│       ├── test_vector_store.py     # Vector store unit tests
│       ├── test_sandbox.py          # Sandbox runner tests
│       ├── test_db.py               # Repository + model tests
│       ├── test_memory.py           # UserMemoryStore tests
│       └── test_api.py              # FastAPI endpoint tests
│
├── spikes/                          # Validated technology spikes
│   ├── spike1_langgraph.py          # LangGraph subgraph + tool calling
│   ├── spike2_qdrant_hybrid.py      # Qdrant sparse + dense hybrid search
│   ├── spike3_langsmith.py          # LangSmith tracing + eval
│   └── spike4_sandbox.py            # Docker sandbox + subprocess
│
├── scripts/
│   ├── generate_benchmark_v1.py     # Benchmark dataset generator
│   └── run_research_demo.py         # Interactive research demo
│
├── data/
│   ├── benchmark/
│   │   ├── benchmark_v1.jsonl       # 120-question evaluation dataset
│   │   └── README.md                # Benchmark documentation
│   ├── concepts/
│   │   └── quant_concepts.yaml      # Quant concept taxonomy
│   └── sample/
│       └── spy_daily.csv            # Sample market data (SPY)
│
└── doc/                             # Technical documentation (Chinese)
    ├── 技术实施方案.md               # Architecture + phase plan
    ├── AI协作分工方案.md             # Workflow + division of labor
    ├── 模块接口契约.md               # Module interface contracts
    ├── 技术预研计划.md               # Spike plan
    └── 渐进性开发记录.md             # Progressive development log
```

---

## Development Guide

### Code Conventions

- **Python 3.11+** with type hints throughout (`TypedDict`, `Annotated`, `Pydantic`)
- **Line length**: 100 characters
- **Docstrings**: Google-style for public functions and classes
- **Imports**: stdlib → third-party → first-party, alphabetized within each group
- **Environment**: All configuration via `os.environ` + `.env` file; no hardcoded credentials

### Key Patterns

1. **Tool decorator** (`@tool` from LangChain) — All T1-T8 tools are decorated LangChain tools with typed input schemas.
2. **Subgraph pattern** — Research and CodeGen are separate `StateGraph` instances that the parent graph routes to based on `router_output`.
3. **HITL with `interrupt()`** — CodeGen calls `interrupt()` after code generation, persisting state to the checkpointer. The API `/resume` endpoint accepts a `Command` with `Resume` to continue.
4. **Mock-first testing** — All external services (LLM, Qdrant, PostgreSQL, Docker) have mock implementations in `conftest.py` fixtures.
5. **Repository pattern** — DB access goes through repository classes (`ApplicationRepo`, `NotesRepo`, `IngestionLogRepo`), not raw SQL.

### Testing

- 21 passing unit tests across 8 test files
- **Mock strategy**: `unittest.mock` for LLM/Qdrant/DB; real FastAPI `TestClient` for API tests
- Run with `pytest tests/ -v` (all tests pass)
- Contract tests (TBD) in `tests/contract/` for cross-module integration

---

## Current Status

**Phase 2 complete (SP3 achieved)** — All core features implemented and tested:

| Deliverable | Status |
|-------------|--------|
| Research subgraph with 7 tools (T1-T3, T5-T8) | Done |
| CodeGen subgraph with T4 + sandbox + HITL | Done |
| Qdrant hybrid search (sparse + dense) | Done |
| Three-tier persistence (Checkpointer + Store + PostgreSQL) | Done |
| FastAPI endpoints (`/chat`, `/interrupts`, `/resume`) | Done |
| 120-question benchmark dataset | Done |
| 21 passing unit tests (8 test files) | Done |
| 4 validated spikes (LangGraph, Qdrant, LangSmith, Sandbox) | Done |

**Phase 3 (starting)** — UI, evaluation pipeline, and production hardening.

---

## Roadmap

| Phase | Focus | Key Deliverables |
|-------|-------|-----------------|
| Phase 0 (done) | Planning | Proposal, architecture design, interface contracts |
| Phase 1 (done) | Foundation | Project structure, config, LLM client, tool skeleton |
| Phase 2 (done) | Core Features | All 8 tools, subgraphs, HITL, hybrid RAG, persistence, tests |
| Phase 3 (next) | UI + Eval | Streamlit UI, automated evaluation pipeline, benchmark scoring |
| Phase 4 | Production | Caching, rate limiting, monitoring, Docker Compose polish |
| Phase 5 | Advanced | Multi-user auth, real-time streaming, model fine-tuning |

---

## License

MIT
