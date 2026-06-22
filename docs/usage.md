# Basic Usage

## 1. Add your AI models

![resources tab](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/resources-tab.png)

The app uses **large language models (LLMs)** and **embedding models** across the QA pipeline. You need at least one of each (or defaults from `flowsettings.py` / `.env`).

Adding more models lets you switch between them in chat and settings.

### Via the UI (recommended)

1. Open the **Resources** tab (hidden in SSO mode; configure models in `flowsettings.py` instead).
2. **LLMs** → **Add**: name, provider (e.g. `ChatOpenAI`), YAML spec, optional **default**.
3. **Embedding Models** → **Add** the same way.
4. Optional: **Reranking models** (e.g. `TeiFastReranking` for a local TEI server — see [README](../README.md)).

Field examples for Ollama (host vs Docker): [`config_example.txt`](../config_example.txt) in the repo root. That file is **not** loaded automatically; copy values into the Resources form.

### Via `.env` and `flowsettings.py`

Copy [`.env.example`](../.env.example) to `.env`. Variables seed defaults in `flowsettings.py` (e.g. `LOCAL_MODEL`, `LOCAL_MODEL_EMBEDDINGS`, `KH_OLLAMA_URL` for Ollama).

<details markdown>
<summary>Provider examples (.env)</summary>

### OpenAI

```shell
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=<your key>
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
```

### Azure OpenAI

```shell
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
```

### Local models (Ollama)

In this fork, `LOCAL_MODEL` and `LOCAL_MODEL_EMBEDDINGS` in `.env` are **Ollama model names**, not file paths. Default LLM/embedding specs in `flowsettings.py` point at `KH_OLLAMA_URL` (default `http://localhost:11434/v1/`).

For a **`.gguf` file** served by llama.cpp, use `scripts/serve_local.py` and set `LOCAL_MODEL` to the **full path** to the file — see [Local models](local_model.md).

</details>

## 2. Upload your documents

![file index tab](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/file-index-tab.png)

1. Open your collection tab (e.g. **File Collection**, or **Files** if multiple index types exist).
2. **Upload and Index** — drag-and-drop or pick files.
3. **File list** — view or delete indexed files.

Supported types are configured per index in `flowsettings.py` (`KH_INDICES` → `supported_file_types`). The default collection includes PDF, Office, images, CSV, HTML, Markdown, ZIP, and more.

## 3. Chat with your documents

![chat tab](https://raw.githubusercontent.com/Cinnamon/kotaemon/main/docs/images/chat-tab.png)

The chat tab has three areas:

1. **Conversation settings** — select/create/rename/delete conversations; choose file context (**Disabled**, **Search All**, or **Select** specific files).
2. **Chat panel** — message input and replies (streaming when the pipeline supports it).
3. **Information panel** — retrieved evidence, citations, and scores.

![information panel](https://raw.githubusercontent.com/Cinnamon/kotaemon/develop/docs/images/info-panel-scores.png)

**Scores (when shown):**

| Score | Meaning |
|-------|---------|
| **Answer confidence** | Model confidence for the answer |
| **Relevance score** | Overall relevance of evidence to the question |
| **Vectorstore score** | Embedding similarity (or full-text label if from keyword search) |
| **LLM relevant score** | LLM-judged relevance |
| **Reranking score** | Cross-encoder / reranker (e.g. Cohere, TEI) |

Generally: `LLM relevant score` > `Reranking score` > vector score. Evidence is ordered by overall relevance and citation presence.

## 4. Reasoning pipelines

In **Settings**, choose a **reasoning** option (registered in `flowsettings.py` → `KH_REASONINGS`), for example:

- `FullQAPipeline` — default RAG with citations
- `FullDecomposeQAPipeline` — decomposed QA
- `ReactAgentPipeline` / `RewooAgentPipeline` — agent-style flows

## 5. Evaluation (optional)

The **Evaluation** tab runs RAG evaluation (e.g. ragas) when configured. Sample questions for `dataset/testing_files` are in `rag_eval_dataset.json` at the repo root.
