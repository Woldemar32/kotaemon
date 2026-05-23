# Kotaemon (custom fork)

<!-- start-intro -->

**Kotaemon** is an open-source RAG application for chatting with your documents (PDF, Office, images, HTML, and more). This repository is a **custom fork** of [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon) with a flattened layout and project-specific defaults.

The codebase ships two Python packages under `src/`:

| Package | Role |
|---------|------|
| **`kotaemon`** | Reusable RAG building blocks: LLMs, embeddings, loaders, vector/doc stores, retrievers, QA pipelines, agents |
| **`ktem`** | Gradio UI: Chat, file collections, Evaluation, Resources, Settings, Help |

- **UI:** Gradio 4 (no separate React frontend; chat is Gradio callbacks, not a public REST API)
- **Python:** 3.11+ (`pyproject.toml`)
- **Config:** `flowsettings.py` + `.env` (see `.env.example`)
- **Runtime data:** `ktem_app_data/` (SQLite, uploads, vector store — do not commit)

For architecture, entry points, and AI-agent routing, see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and [`AI_GUIDE.md`](AI_GUIDE.md).

<!-- end-intro -->

## Quick start (local)

```bash
# From repository root
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements_gerageragera39.txt
pip install -e .
cp .env.example .env            # edit API keys / local model names

python app.py                   # http://localhost:7860
```

Or with **Make** + **uv**:

```bash
make install
make run
```

Default admin login (when user management is enabled): `admin` / `admin` — change after first login.

## Evaluation dataset

Luca's dataset is in `dataset/documents`.

Upload `dataset/testing_files` in the app to try indexing and chat.

`rag_eval_dataset.json` contains simple questions for files in `dataset/testing_files`.

## Docker (Python 3.11)

The container image uses **Python 3.11** and installs dependencies from `requirements_gerageragera39.txt`, matching a local `pip install -r requirements_gerageragera39.txt` setup.

**Prerequisites:** Docker Desktop (or Docker Engine) with Compose v2.

### Docker Compose (recommended)

Persistent app data is stored in **`./ktem_app_data`** on the host (SQLite, uploads, vector index). The container uses **`restart: unless-stopped`**, so it comes back after a reboot; data is kept on `docker compose down` (only removed with `docker compose down -v`).

```bash
cp .env.example .env          # edit keys / model names
docker compose up -d --build  # http://localhost:7860
```

**With Ollama in Docker** (models in volume `ollama_models`):

```bash
# In .env set: KH_OLLAMA_URL=http://ollama:11434/v1/
docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
```

**Optional local reranker (TEI):**

```bash
docker compose --profile reranker up -d
# In Resources use endpoint_url: http://host.docker.internal:8080
```

**Makefile shortcuts:** `make docker-up`, `make docker-up-ollama`, `make docker-down`, `make docker-logs`

**Windows:** `.\scripts\docker-up.ps1 -Build` or `.\scripts\docker-up.ps1 -Ollama -Build`

See [`.env.compose.example`](.env.compose.example) and [`compose.override.example.yml`](compose.override.example.yml) (named volume instead of bind mount).

| Command | Effect |
|---------|--------|
| `docker compose up -d` | Start app, keep data |
| `docker compose down` | Stop containers, **keep** `./ktem_app_data` |
| `docker compose down -v` | Stop and **delete** named volumes (`ollama_models`, etc.) |
| `docker compose logs -f kotaemon` | Follow logs |

### Build images (manual `docker build`)

Three build targets are available:

| Target | Description |
|--------|-------------|
| `lite` | Core app and pinned requirements |
| `full` | Adds OCR, LibreOffice, PyTorch, and document-processing stack |
| `ollama` | `full` plus Ollama and `nomic-embed-text` embedding model |

```bash
# Minimal image
docker build --target lite -t kotaemon:lite .

# Recommended for document RAG (OCR, unstructured, torch)
docker build --target full -t kotaemon:full .

# Includes Ollama for local embeddings/models
docker build --target ollama -t kotaemon:ollama .
```

On **Linux amd64** with an NVIDIA GPU, you can pass a CUDA PyTorch index (optional):

```bash
docker build --target full \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t kotaemon:full .
```

Use a different requirements file (optional):

```bash
docker build --target lite \
  --build-arg REQUIREMENTS_FILE=requirements_gerageragera39.txt \
  -t kotaemon:lite .
```

### Run the app

Create a `.env` file in the project root (copy from `.env.example`) before running, or mount your own env file.

```bash
# Persist app data on the host (Windows cmd)
docker run --rm -p 7860:7860 \
  -v "%cd%\ktem_app_data:/app/ktem_app_data" \
  --env-file .env \
  kotaemon:full
```

```powershell
# PowerShell
docker run --rm -p 7860:7860 `
  -v "${PWD}\ktem_app_data:/app/ktem_app_data" `
  --env-file .env `
  kotaemon:full
```

On Linux/macOS, replace `%cd%` with `$(pwd)`:

```bash
docker run --rm -p 7860:7860 \
  -v "$(pwd)/ktem_app_data:/app/ktem_app_data" \
  --env-file .env \
  kotaemon:full
```

Open **http://localhost:7860** in your browser.

### Local reranker (Text Embeddings Inference)

You can run a **local cross-encoder reranker** in a separate container using [Hugging Face Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference) (TEI). Kotaemon connects to it via the **TeiFastReranking** provider (Resources → Reranking models).

**Prerequisites:** NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you use `--gpus all`.

Start TEI with `BAAI/bge-reranker-v2-m3` on port **8080**:

```bash
docker run -d --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-reranker-v2-m3
```

On first start the image downloads the model; wait until the service responds before running retrieval in Kotaemon.

**Register in Kotaemon**

1. Open the app → **Resources** → **Reranking models** → **Add**.
2. Choose vendor/spec **TeiFastReranking** (see `config_example.txt` for field names).
3. Use this YAML spec (adjust `endpoint_url` if Kotaemon runs in Docker):

```yaml
__type__: kotaemon.rerankings.TeiFastReranking
endpoint_url: http://localhost:8080
is_truncated: true
model_name: BAAI/bge-reranker-v2-m3
```

| Kotaemon runs on | `endpoint_url` |
|------------------|----------------|
| Host (`.venv`, `python app.py`) | `http://localhost:8080` |
| Docker (`kotaemon:full` on same machine) | `http://host.docker.internal:8080` |

Set the model as **default** if you want file-index retrieval to use it automatically (or pick it in index settings where reranking is enabled).

**Stop the reranker container**

```bash
docker ps   # note CONTAINER ID
docker stop <container_id>
```

### Demo / SSO modes

```bash
docker run --rm -p 7860:7860 -e KH_DEMO_MODE=true kotaemon:lite
docker run --rm -p 7860:7860 -e KH_SSO_ENABLED=true --env-file .env kotaemon:lite
```

## Documentation

- **End users:** [docs/usage.md](docs/usage.md), [docs/local_model.md](docs/local_model.md)
- **Developers:** [docs/development/](docs/development/), [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- **MkDocs site:** configure `mkdocs.yml` and run `mkdocs serve` from the repo root

## Legacy install scripts

`scripts/run_*.sh` and `scripts/run_windows.bat` target the **upstream** monorepo layout (`libs/kotaemon`). In this fork, prefer **`pip install -e .`** from the repository root instead.
