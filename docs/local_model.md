# Local LLMs and embedding models

This fork defaults to **Ollama** (OpenAI-compatible API) via `flowsettings.py` and `.env`. You can also use other OpenAI-compatible servers or **llama.cpp** for single `.gguf` files.

!!! note "Docker vs host"
    When Kotaemon runs **inside Docker**, replace `http://localhost` with `http://host.docker.internal` for services on the host machine.

## Ollama (recommended)

1. Install [Ollama](https://github.com/ollama/ollama) and start it.
2. Pull models, for example:

```text
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

3. Set in `.env` (names must match `ollama list`):

```shell
LOCAL_MODEL=qwen2.5:7b
LOCAL_MODEL_EMBEDDINGS=nomic-embed-text
KH_OLLAMA_URL=http://localhost:11434/v1/
```

4. Restart `python app.py`, or add/update models under **Resources** using `config_example.txt` as a field reference.

![Models](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/models.png)

## OpenAI-compatible servers (generic)

In **Resources**, add LLM / embedding specs with `__type__: kotaemon.llms.ChatOpenAI` or `kotaemon.embeddings.OpenAIEmbeddings` and set:

```text
api_key: <provider-specific or dummy>
base_url: http://localhost:<port>/v1/
model: <model id on that server>
```

Examples: [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (often port 5000), other local gateways.

## llama.cpp for a single `.gguf` file

`scripts/serve_local.py` reads `LOCAL_MODEL` from `.env` as a **filesystem path** to a `.gguf` file (not an Ollama name).

```bash
# .env
LOCAL_MODEL=C:\models\my-model.gguf

python scripts/serve_local.py
```

Default server port is **31415** (see `scripts/serve_local.py` and `server_llamacpp_*.bat|sh`).

Register the LLM in **Resources** (OpenAI-compatible):

```text
api_key: dummy
base_url: http://localhost:31415/v1/
model: <name shown by the server>
```

!!! warning "Two meanings of LOCAL_MODEL"
    - **Ollama / flowsettings:** model name in `.env` (e.g. `qwen2.5:7b`).
    - **serve_local.py:** path to `.gguf`. Do not mix these in the same workflow without updating `.env`.

## Local reranker (TEI)

Run [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference) (e.g. `BAAI/bge-reranker-v2-m3` on port 8080) and add **TeiFastReranking** in Resources. See [README — Local reranker](../README.md#local-reranker-text-embeddings-inference).

## Use local models for RAG

1. Set default LLM and embedding in **Resources** (or via `flowsettings.py` defaults).
2. In the file collection settings, set the collection’s embedding model to your local embedding.
3. In **Settings** → retrieval, set **LLM relevant scoring** to a local LLM or disable if the machine cannot handle parallel LLM calls.

![LLM default](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/llm-default.png)

![Index embedding](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/index-embedding.png)

![Retrieval setting](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/retrieval-setting.png)

Start a new conversation to test the pipeline.
