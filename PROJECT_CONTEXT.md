# Project Context

## 1. Overview

**Kotaemon** is an open-source RAG application for chatting with local documents (PDF, Office, images, HTML, etc.). Users upload files into collections, ask questions in chat; the system retrieves relevant chunks, optionally reranks them, and generates answers with citations.

Two Python packages live under `src/`:

- **`kotaemon`** — reusable AI components (LLM, embeddings, loaders, vector/doc stores, retrievers, QA pipelines, agents).
- **`ktem`** (Kotaemon UI) — Gradio app: Chat, file collections, Evaluation, Resources, Settings, Help.

**Audience:** end users (document QA), developers and integrators (`flowsettings.py`, pluggy extensions, `kotaemon` CLI).

**Stack:** **theflow** (declarative pipelines), **LangChain** / **LlamaIndex** for integrations, **SQLModel/SQLite** for app metadata, **Chroma/LanceDB** (and others) for vectors and documents.

This repository is a **custom fork** of [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon): flat `src/kotaemon` and `src/ktem` layout, `requirements_gerageragera39.txt`, evaluation data in `dataset/`. There is no `libs/kotaemon` directory from upstream.

**Core tech:** Python 3.11+, Gradio 4, FastAPI (SSO wrapper only), SQLAlchemy/SQLModel, theflow, pip/uv.

AI quick reference: [`AI_GUIDE.md`](AI_GUIDE.md).

---

## 2. Differences from upstream Kotaemon

| Topic | This fork | Upstream / legacy |
|-------|-----------|-------------------|
| **Sources** | `src/kotaemon/`, `src/ktem/`; `pip install -e .` via `pyproject.toml` | Monorepo `libs/kotaemon/` |
| **CI / scripts** | Some workflows and `run_*.sh` still reference `libs/kotaemon` | Expect that subdirectory |
| **Python** | `requires-python >= 3.11` | Docs sometimes mention 3.10 |
| **Dependencies** | `requirements_gerageragera39.txt` | Standard upstream install paths |

**Practice:** run and test from **repo root**: `pytest tests`, `python app.py`. Verify paths before using `libs/kotaemon`.

---

## 3. Technology stack

| Category | Technologies |
|----------|--------------|
| Language | Python ≥ 3.11 |
| UI | Gradio 4, custom theme (`ktem.assets`), PDF.js, markmap |
| Orchestration | theflow (`BaseComponent`, `Node`, `deserialize`, `flowsettings`) |
| LLM / embeddings | LangChain wrappers, OpenAI-compatible APIs, Ollama, Azure, Cohere, Google, Mistral, VoyageAI, FastEmbed, llama.cpp |
| RAG / indexing | LlamaIndex readers, kotaemon indices, optional GraphRAG / LightRAG / NanoGraphRAG |
| Stores | Chroma, LanceDB, Milvus, Qdrant, Elasticsearch, in-memory, simple file |
| App DB | SQLite (`sqlmodel`), dynamic SQLAlchemy tables per file index |
| Documents | unstructured, PyMuPDF, docling, Azure DI, Adobe PDF (optional) |
| Agents | ReAct, ReWOO, MCP tools |
| Evaluation | ragas (`ktem.evaluation`) |
| CLI | Click + Trogon (`kotaemon` entry point) |
| Package build | hatchling |
| Containers | Docker (`lite` / `full` / `ollama`), `launch.sh` |
| Tests | pytest (minimal coverage) |
| Lint | pre-commit: black, isort, flake8, autoflake, mypy, codespell |

---

## 4. Running the project

### Install

```bash
make install          # uv venv 3.11, requirements_gerageragera39.txt, pip install -e ".[dev]"
# or manually:
python -m venv .venv && pip install -r requirements_gerageragera39.txt && pip install -e .
cp .env.example .env
```

### Dev

```bash
python app.py         # Gradio http://localhost:7860
make run
```

**Local `.gguf` (llama.cpp):**

```bash
# .env: LOCAL_MODEL=C:\path\to\model.gguf  (file path, not Ollama name)
python scripts/serve_local.py   # default port 31415
```

**Docker Compose:** `docker compose up -d --build` — data in `./ktem_app_data`, `restart: unless-stopped`. Optional `--profile ollama`. See `docker-compose.yml`, `README.md`.

**SSO (container via `launch.sh`):**

- `KH_DEMO_MODE=true` → `uvicorn sso_app_demo:app`
- `KH_SSO_ENABLED=true` → `uvicorn sso_app:app` (Gradio at `/app`)

### Tests & lint

```bash
pytest tests
pre-commit run --all-files
```

### CLI

```bash
kotaemon promptui run
kotaemon makedoc kotaemon
kotaemon start-project
```

---

## 5. Project layout

```
├── app.py                 # Gradio launcher
├── flowsettings.py        # Central KH_* config
├── gpu_config.py          # CUDA/DLL (Windows)
├── pyproject.toml
├── requirements_gerageragera39.txt
├── .env.example
├── config_example.txt     # Ollama/TEI examples for Resources UI
├── settings.yaml.example  # Microsoft GraphRAG template
├── launch.sh              # Docker entrypoint
├── Dockerfile
├── Makefile
├── sso_app.py / sso_app_demo.py
├── docs/                  # MkDocs sources
├── scripts/
├── tests/
├── dataset/
├── ktem_app_data/         # Runtime (gitignored)
└── src/
    ├── kotaemon/
    └── ktem/
```

### `src/kotaemon/`

`base/`, `llms/`, `embeddings/`, `rerankings/`, `loaders/`, `storages/`, `indices/`, `agents/`, `contribs/`, `cli.py`.

### `src/ktem/`

`main.py`, `app.py`, `pages/`, `index/`, `reasoning/`, `llms/`, `embeddings/`, `rerankings/`, `db/`, `components.py`, `assets/`, `evaluation/`, `mcp/`, `utils/`.

---

## 6. Entry points

| File | Role |
|------|------|
| `app.py` | `demo.launch()` on :7860 |
| `src/ktem/main.py` | `App` — Gradio tabs |
| `src/ktem/app.py` | `BaseApp.make()` — extensions, indices, reasonings |
| `launch.sh` | Docker: app or SSO |
| `flowsettings.py` | `KH_*`, models, stores, indices |
| `scripts/serve_local.py` | llama.cpp for `.gguf` |

No REST API for main chat — only Gradio.

---

## 7. Architecture

### Layers

1. **UI (Gradio)** — `ktem/pages/*`
2. **Application** — `BaseApp`, `IndexManager`, LLM/embedding managers
3. **Reasoning** — `ktem.reasoning.*` → retrievers + citation QA
4. **Indexing** — `ktem.index.file.*` → load → split → embed → stores
5. **Library** — `kotaemon.*` components
6. **Persistence** — SQLite + `KH_FILESTORAGE_PATH` + Chroma/LanceDB paths

### Chat flow

```
ChatPage.submit → reasoning pipeline → FileIndex retrievers
  → kotaemon.indices.qa → LLM → answer + citations
```

### Indexing flow

```
Upload → FileIndexIndexing → loaders → split → embed
  → vectorstore + docstore → SQL Source/Index tables
```

---

## 8. Key files

| File | Purpose |
|------|---------|
| `flowsettings.py` | Default models, stores, reasonings, feature flags |
| `src/ktem/pages/chat/__init__.py` | Chat logic |
| `src/ktem/reasoning/simple.py` | `FullQAPipeline` |
| `src/ktem/index/file/pipelines.py` | Index + retrieval |
| `src/kotaemon/indices/qa/citation_qa.py` | Citations in answers |
| `src/ktem/db/models.py` | User, Conversation, Settings |

---

## 9. Data models

**SQLModel:** `Conversation`, `User`, `Settings`, `IssueReport`, `Index` (registry).

**Per file index:** dynamic `Source` / `Index` tables `index__{id}__*`.

**Resources:** `LLMTable`, embedding/reranking tables — JSON spec + default flag.

**kotaemon:** `Document`, `RetrievedDocument`, messages, `BaseComponent`.

**Config dicts:** `KH_LLMS`, `KH_EMBEDDINGS`, `KH_RERANKINGS`, `KH_INDICES` with `__type__` for deserialize.

---

## 10. API / routes

- **Chat:** Gradio events only.
- **SSO:** FastAPI mounts Gradio at `/app` (`sso_app.py`).
- **Tabs:** Welcome/Login, Chat, Files (collections), Evaluation, Resources, Settings, Help.

Default port: **7860** (`GRADIO_SERVER_PORT`).

---

## 11. Configuration

| File | Role |
|------|------|
| `flowsettings.py` | Main runtime config |
| `.env` | Provider keys (from `.env.example`) |
| `config_example.txt` | Manual copy-paste for Resources (not auto-loaded) |
| `Dockerfile` | `lite` / `full` / `ollama` |

**`LOCAL_MODEL`:** Ollama **model name** in `flowsettings` / `.env.example`; **file path** when used with `scripts/serve_local.py`.

---

## 12. Tests

- Location: `tests/`
- Run: `pytest tests` from repo root
- CI may still reference `libs/kotaemon` — treat as legacy

---

## 13. Conventions

- Components subclass `kotaemon.base.BaseComponent`.
- Spec dicts use `"__type__": "dotted.path.Class"`.
- Settings prefixed with `KH_*`.
- pluggy: `ktem_declare_extensions`.

---

## 14. Dependency map

```
app.py → ktem.main.App → BaseApp
  → IndexManager → FileIndex → pipelines → kotaemon storages/loaders
  → reasonings ← KH_REASONINGS → FullQAPipeline → citation_qa
  → ChatPage → Conversation (SQLite)

flowsettings + .env → KH_LLMS / KH_EMBEDDINGS → *Manager
```

---

## 15. Known gaps

| Issue | Note |
|-------|------|
| CI `libs/kotaemon` | May fail; use root `pytest tests` |
| `uv.lock` paths | May point to legacy layout |
| `make dev --reload` | `app.py` may not support `--reload` |
| Dual `LOCAL_MODEL` meaning | Name vs path — document which workflow you use |

---

## 16. Quick start for agents

1. [`AI_GUIDE.md`](AI_GUIDE.md)
2. `flowsettings.py`
3. `app.py` → `src/ktem/main.py`
4. Chat: `pages/chat/` + `reasoning/simple.py`
5. Indexing: `index/file/pipelines.py`
6. Run: `pip install -r requirements_gerageragera39.txt && pip install -e . && python app.py`
7. Do not touch `ktem_app_data/` without backup
8. No REST chat API — Gradio only
