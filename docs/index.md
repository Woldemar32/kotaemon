# Getting Started

This documentation describes **this repository** — a custom fork of [Kotaemon](https://github.com/Cinnamon/kotaemon) with sources in `src/kotaemon` and `src/ktem`, Python 3.11+, and configuration via `flowsettings.py` and `.env`.

- **End users:** follow [Basic Usage](usage.md) and [Local models](local_model.md).
- **Developers:** see [Development](development/index.md).

## Prerequisites

- **Python 3.11+**
- An LLM and an embedding model (cloud API or local via Ollama / llama.cpp)
- Optional: Docker for containerized deployment

## Installation (recommended — from source)

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements_gerageragera39.txt
pip install -e .
cp .env.example .env            # set API keys or local model names
python app.py
```

Or with Make and uv:

```bash
make install
make run
```

The app opens at **http://localhost:7860** (port `7860` by default).

When user management is enabled (`KH_FEATURE_USER_MANAGEMENT`, default in `flowsettings.py`), first login is typically **`admin` / `admin`** — change the password in Settings after login.

## Installation (Docker Compose)

```bash
cp .env.example .env
docker compose up -d --build
```

Data persists in `./ktem_app_data` on the host. See [README — Docker Compose](../README.md#docker-compose-recommended).

## Installation (upstream zip / OS scripts)

The original Kotaemon project ships zip releases and `scripts/run_*.bat|sh` installers that install from `libs/kotaemon`. **Those paths do not match this fork.** Prefer `pip install -e .` from this repo root.

If you use upstream’s [Hugging Face Space template](online_install.md), you are deploying Cinnamon’s template, not necessarily this fork’s defaults.

## First launch

1. Open **Resources** and confirm LLM + embedding models (defaults often come from `flowsettings.py` and `.env`).
2. Open your file collection tab (e.g. **File Collection**), upload documents, and click **Upload and Index**.
3. Open **Chat**, select files or **Search All**, and ask a question.

See [Basic Usage](usage.md) for details.

## Help inside the app

The **Help** tab mirrors much of the usage documentation.

## Feedback

Report issues in your project’s issue tracker. Upstream: [Cinnamon/kotaemon issues](https://github.com/Cinnamon/kotaemon/issues).
