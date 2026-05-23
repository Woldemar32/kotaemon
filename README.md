Luca's dataset is in `.\dataset\documents`

Upload `.\dataset\testing_files` in your system!

`rag_eval_dataset.json` is a file with simple questions for files in `.\dataset\testing_files`

## Docker (Python 3.11)

The container image uses **Python 3.11** and installs dependencies from `requirements_gerageragera39.txt`, matching a local `pip install -r requirements_gerageragera39.txt` setup.

**Prerequisites:** Docker Desktop (or Docker Engine) running, with BuildKit enabled (default in recent Docker versions).

### Build images

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
# Persist app data on the host
docker run --rm -p 7860:7860 \
  -v "%cd%\ktem_app_data:/app/ktem_app_data" \
  --env-file .env \
  kotaemon:full

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


docker run -d --gpus all -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:86-1.9 --model-id BAAI/bge-reranker-v2-m3
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

### Local install (without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements_gerageragera39.txt
pip install -e .
python app.py
```

Python **3.11** is required (`requires-python >= 3.11` in `pyproject.toml`).
