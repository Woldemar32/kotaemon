# Basic Usage

## 1. Add your AI models

![resources tab](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/resources-tab.png)

Provide at least one **LLM** and one **embedding model** (via **Resources**, or defaults from `.env` / `flowsettings.py`).

1. **Resources** → **LLMs** → **Add** (name, provider, YAML spec, optional default).
2. **Resources** → **Embedding Models** → **Add**.
3. Optional: **Reranking models** (e.g. TeiFastReranking for a local TEI server).

See `config_example.txt` in the repo root for Ollama field examples (host vs Docker).

<details markdown>
<summary>(Optional) Configure via .env</summary>

Copy `.env.example` to `.env`. For **Ollama**, set model **names**:

```shell
LOCAL_MODEL=qwen2.5:7b
LOCAL_MODEL_EMBEDDINGS=nomic-embed-text
```

For a **`.gguf` file**, use `scripts/serve_local.py` with `LOCAL_MODEL` set to the **file path** — not the same as the Ollama name workflow.

### OpenAI

```shell
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=<your key>
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
```

</details>

## 2. Upload your documents

Open your file collection tab (e.g. **File Collection**), upload files, and click **Upload and Index**. Manage files in the file list section.

## 3. Chat with your documents

Use **Conversation settings** to pick files (**Disabled**, **Search All**, or **Select**). Ask questions in the chat panel; evidence and citations appear in the **Information** panel.

**Scores:** answer confidence, relevance, vectorstore, LLM relevance, reranking — when enabled by the pipeline.
