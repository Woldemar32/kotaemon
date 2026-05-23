# AI Guide for this repository

Short instructions for Codex, Cursor, and other AI agents. Read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) first, then only files relevant to the task.

---

## What this is

**Kotaemon fork** — RAG app for chatting with documents (PDF, Office, images, etc.). UI is **Gradio** (`ktem`); reusable RAG library is **`kotaemon`** in `src/`. No separate React frontend; the main API is Gradio callbacks, not REST.

Source layout: `src/kotaemon/`, `src/ktem/` (not upstream `libs/kotaemon`). Python **≥ 3.11** (`pyproject.toml`).

---

## Most important files

| Task | Read first |
|------|------------|
| Launch app | `app.py`, `launch.sh`, `Makefile` |
| Models & indices config | `flowsettings.py`, `.env.example`, `config_example.txt` |
| Chat UI & events | `src/ktem/pages/chat/__init__.py`, `src/ktem/main.py` |
| Reasoning / RAG pipeline | `src/ktem/reasoning/simple.py`, `src/kotaemon/indices/qa/citation_qa.py` |
| File indexing | `src/ktem/index/file/pipelines.py`, `src/ktem/index/file/index.py` |
| LLM providers | `src/kotaemon/llms/`, `src/ktem/llms/manager.py`, `flowsettings.KH_LLMS` |
| Embeddings | `src/kotaemon/embeddings/`, `src/ktem/embeddings/manager.py` |
| Database | `src/ktem/db/models.py`, `src/ktem/db/engine.py`, `src/ktem/index/models.py` |
| Docker | `Dockerfile`, `launch.sh`, `README.md` |
| Tests / CI | `tests/`, `.github/workflows/unit-test.yaml`, `pyproject.toml` |
| App shell | `src/ktem/app.py`, `src/ktem/components.py` |
| Local `.gguf` (llama.cpp) | `scripts/serve_local.py`, `scripts/server_llamacpp_*.bat/sh` |
| GraphRAG / LightRAG | `flowsettings.py` (`USE_*` flags), `src/ktem/index/file/graph/` |

---

## Usually skip unless the task requires them

| Path | Reason |
|------|--------|
| `ktem_app_data/` | Runtime: SQLite, uploads, vectorstore, HF cache |
| `uv.lock` | Large lock; may reference legacy `libs/kotaemon` |
| `__pycache__/`, `.venv/` | Cache / environment |
| `docs/theme/assets/` | Generated MkDocs assets |
| `.omx/` | Internal state files |
| `dataset/` | Test documents, not application code |
| `templates/project-default/` | Cookiecutter template |
| Adobe / Azure DI / MS GraphRAG | Heavy optional integrations |
| All of `src/kotaemon/loaders/` | Only when changing parsing for a specific format |

---

## Common tasks

| Task | Where to go |
|------|-------------|
| Change chat behavior | `src/ktem/pages/chat/`, `src/ktem/reasoning/` |
| Change prompts / answer format | `src/kotaemon/indices/qa/`, `src/kotaemon/llms/prompts/`, `src/ktem/reasoning/prompt_optimization/` |
| Change document indexing | `src/ktem/index/file/pipelines.py`, `src/kotaemon/indices/ingests/` |
| Add Ollama / local model | `.env`, `flowsettings.py` (`LOCAL_MODEL`, `KH_OLLAMA_URL`), UI `src/ktem/llms/`, `config_example.txt` |
| Run llama.cpp for `.gguf` | `scripts/serve_local.py` (`LOCAL_MODEL` = **path to file**), then add LLM in Resources with `base_url` pointing to port **31415** |
| New LLM provider | `src/kotaemon/llms/`, entry in `KH_LLMS`, `src/ktem/llms/ui.py` + `db.py` |
| Fix Docker | `docker-compose.yml`, `Dockerfile`, `launch.sh`, `README.md`, `make docker-up` |
| Fix tests / CI | `tests/`, `.github/workflows/unit-test.yaml` — run `pytest tests` from repo root, Python 3.11+ |
| GraphRAG / LightRAG | `flowsettings.GRAPHRAG_INDEX_TYPES`, `src/ktem/index/file/graph/`, `settings.yaml.example` |

---

## Commands

```bash
# Install (from repo root)
make install
# or: python -m venv .venv && pip install -r requirements_gerageragera39.txt && pip install -e .

# Run (Gradio :7860)
python app.py

# Local llama.cpp for .gguf (path in .env)
python scripts/serve_local.py

# Docker Compose (persistent ./ktem_app_data, restart unless-stopped)
docker compose up -d --build
docker compose --profile ollama up -d --build
docker compose down          # keeps data

# Tests
pytest tests

# Lint (if pre-commit is configured)
pre-commit run --all-files
```

---

## Warnings

1. **Never commit or print** `.env` contents — only variable names from `.env.example`.
2. **Do not delete or edit** `ktem_app_data/` without a backup — user DB and indices live there.
3. **Do not trust paths** `libs/kotaemon` in CI, `uv.lock`, `run_*.sh`, `mkdocs.yml` without verifying — this fork uses `src/` (see §2 in `PROJECT_CONTEXT.md`).
4. **`LOCAL_MODEL`**: in `flowsettings`/Ollama — **model name**; in `serve_local.py` — **path to `.gguf`**. Different workflows.
5. **`config_example.txt`** is not loaded by the app — example fields for the Resources tab.
6. Before broad edits: **`PROJECT_CONTEXT.md` → 2–5 relevant files** — do not scan the whole repo.
