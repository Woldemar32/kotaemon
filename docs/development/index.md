# Development

This section is for contributors and integrators working on **`kotaemon`** (library) and **`ktem`** (Gradio app) in `src/`.

## Repository layout

| Path | Package | Role |
|------|---------|------|
| `src/kotaemon/` | `kotaemon` | RAG components: LLM, embeddings, loaders, stores, indices, agents |
| `src/ktem/` | `ktem` | Application UI, index manager, reasoning pipelines, DB |
| `flowsettings.py` | — | Runtime config (`KH_*`, default models, stores) |
| `app.py` | — | Inserts `src/` on `sys.path`, launches Gradio |
| `tests/` | — | pytest from **repository root** |

Read [`PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md) for architecture, data flow, and configuration. [`AI_GUIDE.md`](../../AI_GUIDE.md) lists task → file mappings for agents.

## Setup

```bash
git clone <your-repo-url>
cd kotaemon

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -r requirements_gerageragera39.txt
pip install -e ".[dev]"
pre-commit install
pytest tests
python app.py
```

Python **3.11+** is required. Do **not** use `cd libs/kotaemon` — that path is upstream-only.

## Topics in this section

- [Contributing](contributing.md) — PR workflow, conventions, CI notes
- [Creating a component](create-a-component.md) — `BaseComponent` and theflow
- [Customize flow logic](../pages/app/customize-flows.md) — register reasoning/index pipelines
- [File index](../pages/app/index/file.md) — indexing and retrieval hooks
- [Settings](../pages/app/settings/overview.md) — developer vs user settings
- [User management extension](../pages/app/ext/user-management.md)
- [Utilities](utilities.md) — PromptUI, CLI, scaffolding

## API reference

MkDocs can generate API pages from `src/kotaemon` (see `mkdocs.yml` and `docs/scripts/generate_reference_docs.py`). Build with:

```bash
mkdocs serve
```

## Extending the app

1. Implement pipelines as `kotaemon.base.BaseComponent` subclasses (or `ktem` index/reasoning base classes).
2. Register them in `flowsettings.py` (`KH_REASONINGS`, `KH_INDICES`, `KH_INDEX_TYPES`, optional `FILE_INDEX_PIPELINE`).
3. Expose user-tunable fields via `get_user_settings()` / `get_info()` where applicable.

Pluggy entry points (`ktem_declare_extensions`) allow extensions without forking core UI code.
