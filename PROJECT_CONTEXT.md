# Project Context

## 1. Краткое описание проекта

**Kotaemon** — open-source RAG-приложение для чата с локальными документами (PDF, Office, изображения, HTML и др.). Пользователь загружает файлы в коллекции, задаёт вопросы в чате; система извлекает релевантные фрагменты, ранжирует их и генерирует ответ с цитатами.

Проект состоит из двух Python-пакетов в `src/`:
- **`kotaemon`** — библиотека переиспользуемых AI-компонентов (LLM, embeddings, loaders, vector/doc stores, retrievers, QA-пайплайны, агенты).
- **`ktem`** (Kotaemon UI) — Gradio-приложение: вкладки Chat, Files/индексы, Evaluation, Resources, Settings, Help.

Целевая аудитория: конечные пользователи (документный QA), разработчики и интеграторы (кастомизация через `flowsettings.py`, расширения pluggy, CLI `kotaemon`).

Техническая цель: гибкий RAG-стек на базе **theflow** (декларативные пайплайны из dotted-path классов), **LangChain** / **LlamaIndex** для части интеграций, **SQLModel/SQLite** для метаданных приложения, **Chroma/LanceDB** и др. для векторов и документов.

В этой копии репозитория README указывает на датасет в `dataset/` и кастомный `requirements_gerageragera39.txt`; структура исходников — плоская (`src/kotaemon`, `src/ktem`), без каталога `libs/kotaemon` из upstream.

Основные технологии: **Python 3.11+**, **Gradio 4**, **FastAPI** (только SSO-обёртка), **SQLAlchemy/SQLModel**, **theflow**, **uv/pip** для зависимостей.

Подробная шпаргалка для AI-агентов: [`AI_GUIDE.md`](AI_GUIDE.md).

---

## 2. Отличия этого форка от upstream Kotaemon

Эта копия репозитория — **форк/кастомизация** [Cinnamon/kotaemon](https://github.com/Cinnamon/kotaemon), а не byte-to-byte клон upstream. Перед правками CI, установщиков или путей в документации сверяйте layout с фактическим деревом файлов.

| Тема | В этом форке | Upstream / legacy |
|------|--------------|-------------------|
| **Исходники** | `src/kotaemon/` и `src/ktem/` в корне; `pip install -e .` через `pyproject.toml` (`[tool.hatch.build.targets.wheel] packages`) | Монорепо с `libs/kotaemon/` |
| **CI и скрипты** | `.github/workflows/unit-test.yaml` делает `cd libs/kotaemon && pytest`; `scripts/run_*.sh`, `update_*.bat`, `mkdocs.yml`, `uv.lock` могут ссылаться на `libs/kotaemon` | Ожидают subdirectory `libs/kotaemon` |
| **Python** | `pyproject.toml`: `requires-python = ">=3.11"` | Документация upstream иногда упоминает 3.10 |
| **GitHub Actions unit-test** | Matrix `python-version: ["3.10", "3.11"]` + `cd libs/kotaemon` — **может падать** в этой структуре (нет `libs/kotaemon`, 3.10 ниже минимума пакета) | Рассчитано на старый layout |
| **Зависимости** | `requirements_gerageragera39.txt`, кастомный README (dataset, Docker) | Стандартный `libs/kotaemon` install в contrib-доках |

**Практика для разработчика/AI:** локальный запуск и тесты — из **корня**: `pytest tests`, `python app.py`. Пути вида `libs/kotaemon` не использовать без проверки `Test-Path` / `ls`.

---

## 3. Технологический стек

| Категория | Технологии |
|-----------|------------|
| Язык | Python ≥ 3.11 |
| UI | Gradio 4, кастомная тема (`ktem.assets`), JS (PDF.js viewer, markmap) |
| Оркестрация компонентов | theflow (`BaseComponent`, `Node`, `deserialize`, `flowsettings`) |
| LLM / embeddings | LangChain-обёртки, OpenAI-совместимые API, Ollama, Azure OpenAI, Cohere, Google, Mistral, VoyageAI, FastEmbed, llama.cpp |
| RAG / индексация | LlamaIndex readers, kotaemon indices (VectorIndexing, VectorRetrieval), GraphRAG (Microsoft), LightRAG, NanoGraphRAG |
| Векторные / doc stores | Chroma, LanceDB, Milvus, Qdrant, Elasticsearch, in-memory, simple file |
| БД приложения | SQLite (`sqlmodel`), динамические SQLAlchemy-таблицы для file index |
| Документы | unstructured, PyMuPDF, docling, Azure Document Intelligence, Adobe PDF Services (опционально) |
| Агенты | ReAct, ReWOO, MCP tools |
| Оценка качества | ragas (`ktem.evaluation`) |
| CLI | Click + Trogon TUI, entry point `kotaemon` |
| Сборка пакета | hatchling (`pyproject.toml`) |
| Контейнеризация | Docker (targets: lite / full / ollama), `launch.sh` |
| Деплой | fly.io (`fly.toml`), GitHub Actions |
| Тесты | pytest (минимальное покрытие) |
| Линтинг / формат | pre-commit: black, isort, flake8, autoflake, mypy, codespell, prettier (md/yaml) |
| Внешние API (по конфигу) | OpenAI, Azure OpenAI, Cohere, Google, Mistral, Groq, VoyageAI, Tavily web search, DuckDuckGo, Wikipedia, Ollama, Keycloak/Google OAuth (SSO) |

---

## 4. Как запустить проект

### Установка зависимостей

**Рекомендуемый путь (Makefile + uv):**

```bash
make install          # uv venv .venv --python 3.11, pip install -r requirements_gerageragera39.txt, pip install -e ".[dev]"
make install-gpu      # + torch CUDA (cu121)
```

**Вручную (из README):**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements_gerageragera39.txt
pip install -e .
```

Скопировать `.env.example` → `.env` и задать ключи провайдеров (без коммита секретов).

### Dev-режим

```bash
python app.py                   # Gradio на порту 7860 (по умолчанию), inbrowser=True
make run                        # .venv/Scripts/python app.py (Windows в Makefile)
make dev                        # то же с --reload (если поддерживается launcher)
```

**Локальная `.gguf` (llama.cpp, опционально):**

```bash
# В .env: LOCAL_MODEL=C:\path\to\model.gguf  (путь к файлу, не имя Ollama)
python scripts/serve_local.py
```

**Docker:**

```bash
docker build --target full -t kotaemon:full .
docker run --rm -p 7860:7860 -v "%cd%\ktem_app_data:/app/ktem_app_data" --env-file .env kotaemon:full
```

**SSO / demo (через `launch.sh` в контейнере):**

- `KH_DEMO_MODE=true` → `uvicorn sso_app_demo:app`
- `KH_SSO_ENABLED=true` → `uvicorn sso_app:app` (Gradio на `/app`)

### Сборка

- Python-пакет: `pip install -e .` или hatch wheel из `pyproject.toml`
- Docker: `docker build --target lite|full|ollama`

### Тесты

```bash
pytest tests                  # документировано в docs/development/contributing.md
```

*Предположение:* CI в `.github/workflows/unit-test.yaml` выполняет `cd libs/kotaemon && pytest` — в **этой** копии репозитория каталога `libs/kotaemon` нет; локально используйте `pytest tests` из корня.

### Линтинг / форматирование

```bash
pre-commit install
pre-commit run --all-files
```

Отдельно: `black`, `isort`, `flake8`, `mypy` (см. `.pre-commit-config.yaml`). В `pyproject.toml` указаны `ruff` в `[dev]`, но pre-commit использует flake8/black.

### CLI

```bash
kotaemon promptui run          # Gradio из promptui.yml
kotaemon makedoc kotaemon      # генерация docs
kotaemon start-project         # cookiecutter шаблон
```

### GPU (Windows)

```bash
python gpu_config.py --check
```

Вызывается из `app.py` через `configure_gpu_for_project()`.

---

## 5. Структура проекта

```
kotaemon-src/
├── app.py                 # Главный launcher Gradio
├── flowsettings.py        # Центральный конфиг приложения (theflow settings)
├── gpu_config.py          # CUDA/DLL для Windows
├── pyproject.toml         # Метаданные пакета, зависимости, scripts
├── requirements_gerageragera39.txt  # Pin-зависимости (Docker/local)
├── uv.lock
├── .env.example           # Шаблон переменных провайдеров
├── config_example.txt     # Примеры Ollama/TEI для UI Resources (.venv и Docker)
├── settings.yaml.example  # Шаблон Microsoft GraphRAG
├── launch.sh              # Entrypoint Docker (app / SSO / demo)
├── Dockerfile
├── Makefile
├── sso_app.py             # FastAPI + gradiologin (prod SSO)
├── sso_app_demo.py        # Demo без user management
├── mkdocs.yml             # Документация сайта
├── docs/                  # MkDocs исходники
├── scripts/               # run_*, update_*, migrate, rag eval, pdfjs
├── templates/             # cookiecutter project-default
├── tests/                 # pytest (1 файл)
├── dataset/               # Документы для eval (упомянуты в README)
├── ktem_app_data/         # Runtime data (SQLite, files, caches) — не коммитить
├── src/
│   ├── kotaemon/          # RAG-библиотека
│   └── ktem/              # Gradio UI
└── .github/workflows/     # CI: unit-test, docker, release
```

### `src/kotaemon/` — библиотека компонентов

| Подпапка | Назначение |
|----------|------------|
| `base/` | `BaseComponent`, `Document`, сообщения LLM, схемы |
| `llms/` | Chat/completion LLM, prompts, OpenAI/Azure/Ollama/… |
| `embeddings/` | Embedding-провайдеры |
| `rerankings/` | Cohere, VoyageAI rerank |
| `loaders/` | PDF, DOCX, HTML, OCR, Azure DI, unstructured |
| `parsers/` | Постобработка текста |
| `storages/` | `docstores/`, `vectorstores/` (Chroma, LanceDB, …) |
| `indices/` | ingest, splitters, retrievers, rankings, QA (`citation_qa`) |
| `agents/` | ReAct, ReWOO, tools (в т.ч. MCP) |
| `chatbot/` | Простые respondent-обёртки |
| `contribs/` | promptui, docs generator |
| `cli.py` | Click CLI |

Связь: импортируется из `ktem` через dotted paths в `flowsettings` и пайплайны reasoning/index.

### `src/ktem/` — UI-приложение

| Подпапка | Назначение |
|----------|------------|
| `main.py` | `App(BaseApp)` — вкладки Gradio |
| `app.py` | `BaseApp`: тема, индексы, reasoning, pluggy extensions |
| `pages/` | Chat, Settings, Resources, Evaluation, Login, Setup, Help |
| `index/` | `IndexManager`, `FileIndex`, GraphRAG-варианты, pipelines |
| `reasoning/` | QA-пайплайны (simple, react, rewoo) |
| `llms/`, `embeddings/`, `rerankings/` | UI + `*Manager` + SQLite tables |
| `db/` | SQLModel models, engine |
| `components.py` | docstore/vectorstore factories, `reasonings` registry |
| `assets/` | CSS, JS, icons, PDF.js |
| `evaluation/` | ragas eval UI logic |
| `mcp/` | MCP integration UI |
| `utils/` | render, rate limit, conversation helpers |

### `docs/`

MkDocs: установка, usage, development (компоненты, customize flows). Не путать с runtime `KH_DOC_DIR` в flowsettings.

### `scripts/`

Установщики ОС (`run_windows.bat`, …), обновления, миграция Chroma, `run_rag_eval.py`, скачивание PDF.js.

- **`serve_local.py`** — интерактивный запуск **llama.cpp** HTTP-сервера для локальной **`.gguf`**-модели: читает из `.env` переменную `LOCAL_MODEL` как **путь к файлу** (не имя Ollama-модели), при подтверждении вызывает `server_llamacpp_*.bat/sh` (порт по умолчанию 31415, эвристика `chat_format` для Qwen).
- **`config_example.txt`** — не исполняемый конфиг; шпаргалка полей для добавления моделей в UI **Resources**: блоки для **`.venv`** (`base_url: http://localhost:11434/v1`) и **Docker** (`http://host.docker.internal:11434/v1`), плюс пример **TEI** reranker (`TeiFastReranking`, порт 8080).

### `ktem_app_data/`

Создаётся при первом запуске: `user_data/sql.db`, `files/`, `vectorstore/`, `docstore/`, HuggingFace cache, gradio tmp.

### Не анализировать подробно

`__pycache__`, `.venv`, `node_modules` (нет в проекте), автоген в `ktem_app_data`, `.omx/`.

---

## 6. Главные entry points

| Файл | Что запускает | Подключает |
|------|---------------|------------|
| `app.py` | Gradio `demo.launch()` на :7860 | `gpu_config`, `theflow.settings`, `ktem.main.App`, `app.make()` |
| `src/ktem/main.py` | Класс `App`: вкладки UI | `ChatPage`, `SettingsPage`, `IndexManager`, `flowsettings` flags |
| `src/ktem/app.py` | `BaseApp.make()` → `gr.Blocks` | pluggy extensions, `IndexManager`, reasonings из `KH_REASONINGS` |
| `launch.sh` | Docker: `python app.py` или uvicorn SSO | `sso_app` / `sso_app_demo` / Ollama sidecar |
| `sso_app.py` | FastAPI + `grlogin.mount_gradio_app(..., "/app")` | Google/Keycloak OAuth, тот же `App` |
| `sso_app_demo.py` | Упрощённый FastAPI demo | `KH_DEMO_MODE`, без user mgmt |
| `src/kotaemon/cli.py` | `kotaemon` CLI (`main`, `promptui`, `makedoc`, `start_project`) | promptui, cookiecutter |
| `flowsettings.py` | Конфиг theflow (не исполняемый) | `KH_*`, `KH_LLMS`, `KH_INDICES`, stores, reasoning list |
| `scripts/serve_local.py` | llama.cpp server для `.gguf` по пути из `LOCAL_MODEL` в `.env` | `server_llamacpp_windows.bat` / `_linux.sh` / `_macos.sh` |
| `scripts/run_rag_eval.py` | Оценка RAG на `rag_eval_dataset.json` | dataset |

Отдельного React/Vue frontend и REST API для основного чата **нет** — взаимодействие через Gradio events.

---

## 7. Архитектура проекта

### Слои

1. **UI (Gradio)** — `ktem/pages/*`, события `.click()` / `.submit()`, состояние `gr.State`.
2. **Application** — `BaseApp`, `IndexManager`, managers LLM/embeddings/rerank.
3. **Reasoning** — `ktem.reasoning.*` наследуют `BaseReasoning`, собирают retrievers + LLM + citation QA.
4. **Indexing** — `ktem.index.file.*` pipelines: load → split → embed → vector/doc store.
5. **Library** — `kotaemon.*` переиспользуемые `BaseComponent` пайплайны.
6. **Persistence** — SQLite (users, conversations, LLM specs, index registry) + файлы в `KH_FILESTORAGE_PATH` + Chroma/LanceDB paths.

### Поток данных (типичный вопрос в чате)

```
ChatPage.submit
  → выбор reasoning pipeline (из components.reasonings)
  → retrievers FileIndex (VectorRetrieval + rerank)
  → kotaemon.indices.qa (PrepareEvidence, AnswerWithContext)
  → LLM (llms.manager)
  → ответ + citations в UI
```

### Поток индексации файла

```
File index UI upload
  → FileIndexIndexing pipeline (pipelines.py)
  → kotaemon loaders/extractors
  → TokenSplitter → embeddings
  → vectorstore + docstore (get_vectorstore/get_docstore)
  → SQL Source/Index tables per index id
```

### Взаимодействие модулей

- `flowsettings` → при старте заполняет `KH_LLMS`, `KH_EMBEDDINGS`, `KH_INDICES`; managers синхронизируют с SQLite.
- `theflow.utils.modules.deserialize` / `import_dotted_string` — инстанцирование классов по `__type__` в spec.
- **pluggy** (`ktem` entry points) — расширения reasoning/index без форка.
- **LangChain/LlamaIndex** — внутри конкретных loader/LLM/retriever реализаций, не как единый фасад UI.

---

## 8. Ключевые файлы и их назначение

| Файл | Назначение | Почему важен |
|------|------------|------------|
| `app.py` | Запуск Gradio | Единственная точка для локального run |
| `flowsettings.py` | Все `KH_*` настройки, модели, индексы | Меняют поведение без правки UI |
| `src/ktem/main.py` | Компоновка вкладок | Карта UI приложения |
| `src/ktem/app.py` | `BaseApp`, lifecycle Gradio | Регистрация reasoning, extensions |
| `src/ktem/components.py` | Doc/vector store, `reasonings` dict | Фабрики хранилищ |
| `src/ktem/pages/chat/__init__.py` | `ChatPage` — основная логика чата | Самый большой UI-модуль |
| `src/ktem/reasoning/simple.py` | `FullQAPipeline` и варианты | Дефолтный RAG QA |
| `src/ktem/index/manager.py` | CRUD индексов | Связь UI ↔ SQLite Index |
| `src/ktem/index/file/index.py` | `FileIndex` | Ядро file-based RAG |
| `src/ktem/index/file/pipelines.py` | Indexing/retrieval pipelines | Ingest + search |
| `src/kotaemon/indices/qa/citation_qa.py` | Ответ с цитированием | Качество ответов |
| `src/kotaemon/base/schema.py` | `Document`, messages | Общая модель данных |
| `src/ktem/db/models.py` | User, Conversation, Settings | Персистентность чата |
| `src/ktem/llms/manager.py` | Пул LLM из БД + flowsettings | Выбор модели в чате |
| `pyproject.toml` | Зависимости, `kotaemon` script | Сборка и CLI |
| `requirements_gerageragera39.txt` | Pinned deps | Docker и reproducible install |
| `Dockerfile` / `launch.sh` | Prod-образ и режимы | Деплой |
| `.env.example` | Имена env для провайдеров | Интеграции без чтения секретов |
| `sso_app.py` | SSO-обёртка | Enterprise auth |
| `config_example.txt` | Примеры Ollama/TEI для Resources | Локальный/Docker Ollama без копания в docs |
| `scripts/serve_local.py` | Старт llama.cpp для `.gguf` | Альтернатива Ollama для `LOCAL_MODEL`-path |

---

## 9. Основные сущности и модели данных

### SQLModel / SQLite (`src/ktem/db/`)

| Модель | Поля (высокий уровень) | Где используется |
|--------|------------------------|------------------|
| `Conversation` | `id`, `name`, `user`, `is_public`, `data_source` (JSON: messages, files), timestamps | `ChatPage`, history |
| `User` | `username`, `password` (hash), `admin` | Login, RBAC вкладок |
| `Settings` | `user`, `setting` JSON | Пользовательские настройки |
| `IssueReport` | отчёты о проблемах | `ReportIssue` в чате |
| `Index` (`ktem/index/models.py`) | `name`, `index_type`, `config` | `IndexManager` |

File index создаёт **динамические** SQLAlchemy-модели `Source`, `Index` с именами `index__{id}__*`.

### Таблицы ресурсов

`LLMTable`, embedding/reranking tables в `src/ktem/llms/db.py`, `embeddings/db.py`, `rerankings/db.py` — spec JSON + default flag.

### kotaemon схемы (`src/kotaemon/base/`)

| Тип | Назначение |
|-----|------------|
| `Document` | Текстовый chunk + metadata |
| `RetrievedDocument` | Document + score для retrieval |
| `HumanMessage` / `AIMessage` / `SystemMessage` | LLM диалог |
| `BaseComponent` | Узел theflow-пайплайна с `run()` |

### Конфигурационные dict (не ORM)

- `KH_LLMS`, `KH_EMBEDDINGS`, `KH_RERANKINGS` — spec с `__type__` для deserialize.
- `KH_INDICES` — список коллекций при первом setup.

---

## 10. API / Routes / Endpoints

### Основное приложение

**Нет публичного REST API** для чата. API — это Gradio callbacks внутри `pages/*`.

### FastAPI (только SSO-режим)

| Метод | Путь | Назначение | Файл |
|-------|------|------------|------|
| GET | `/favicon.ico` | Favicon | `sso_app.py` |
| * | `/app/*` | Gradio UI (mount) | `grlogin.mount_gradio_app` |

`sso_app_demo.py`: `/`, `/login`, `/logout`, `/auth` — упрощённая auth-страница для demo.

### Gradio «маршруты» (вкладки)

| Вкладка | Модуль | Функция |
|---------|--------|---------|
| Welcome / Login | `pages/login.py` | Auth (если `KH_FEATURE_USER_MANAGEMENT`) |
| Chat | `pages/chat/` | Q&A, файлы в сессии, web search command |
| File / Graph collections | `index/file/ui.py`, graph UIs | Upload, index management |
| Evaluation | `pages/evaluation.py` | RAG eval (ragas) |
| Resources | `pages/resources/` | CRUD LLM/embeddings |
| Settings | `pages/settings.py` | App/reasoning/index settings |
| Help | `pages/help.py` | Markdown help |

Порт по умолчанию: **7860** (`GRADIO_SERVER_PORT`).

---

## 11. Конфигурация

| Файл | Назначение |
|------|------------|
| `pyproject.toml` | Пакет `kotaemon` 0.1.0, зависимости, `[project.scripts]`, hatch packages `src/kotaemon`, `src/ktem` |
| `requirements_gerageragera39.txt` | Полный pin для Docker/local |
| `uv.lock` | Lock для uv (editable path может указывать на `libs/kotaemon` — см. §2, §15) |
| `flowsettings.py` | **Главный runtime-конфиг**: пути данных, feature flags, stores, indices, reasonings |
| `.env` / `.env.example` | API keys и endpoints провайдеров (не коммитить `.env`) |
| `settings.yaml.example` | Microsoft GraphRAG (при `USE_CUSTOMIZED_GRAPHRAG_SETTING`) |
| `config_example.txt` | Примеры полей Ollama LLM/embeddings для `.venv` и Docker (`host.docker.internal`); опционально TEI reranker — копировать в форму **Resources**, не подхватывается автоматически |
| `Dockerfile` | multi-stage: lite → full → ollama |
| `fly.toml` | Fly.io: порт 7860, volume `ktem_app_data` |
| `.pre-commit-config.yaml` | Хуки качества кода |
| `mkdocs.yml` | Док-сайт (paths могут указывать на upstream `libs/kotaemon`) |

### Важные переменные окружения (только назначение)

| Переменная | Назначение |
|------------|------------|
| `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDINGS_MODEL` | OpenAI LLM/embeddings |
| `AZURE_OPENAI_*` | Azure OpenAI |
| `COHERE_API_KEY`, `MISTRAL_API_KEY`, `VOYAGE_API_KEY`, `GOOGLE_API_KEY` | Другие провайдеры |
| `LOCAL_MODEL`, `LOCAL_MODEL_EMBEDDINGS`, `KH_OLLAMA_URL` | Ollama: в `flowsettings` — **имя модели**; в `serve_local.py` — **путь к `.gguf`** (другое использование той же переменной) |
| `GRAPHRAG_*`, `USE_CUSTOMIZED_GRAPHRAG_SETTING` | Microsoft GraphRAG |
| `USE_LIGHTRAG`, `USE_NANO_GRAPHRAG`, `USE_MS_GRAPHRAG`, `USE_GLOBAL_GRAPHRAG` | Типы graph-индексов |
| `KH_DEMO_MODE`, `KH_SSO_ENABLED`, `KH_GRADIO_SHARE` | Режимы приложения |
| `KH_FEATURE_USER_MANAGEMENT`, `KH_FEATURE_USER_MANAGEMENT_ADMIN/PASSWORD` | Локальные пользователи |
| `AUTHENTICATION_METHOD`, `GOOGLE_CLIENT_*`, `KEYCLOAK_*` | SSO |
| `GRADIO_SERVER_PORT`, `GRADIO_TEMP_DIR`, `GR_FILE_ROOT_PATH` | Сервер и пути файлов |
| `PDF_SERVICES_CLIENT_*`, `AZURE_DI_*` | Adobe/Azure document parsing |
| `OLLAMA_ENABLED` | Запуск ollama serve в Docker |

---

## 12. Тесты

| Аспект | Детали |
|--------|--------|
| Расположение | `tests/test_information_panel_ordering.py` |
| Framework | pytest (`pytest.importorskip("theflow")`) |
| Запуск | `pytest tests` из корня репозитория |
| Покрытие | Один unit-тест: порядок citation panel в `AnswerWithContextPipeline` при нулевых LLM scores |
| CI | `.github/workflows/unit-test.yaml` — `uv sync`, но шаг `cd libs/kotaemon` **не соответствует** структуре этой копии |

Важные сценарии **не** покрыты автотестами: indexing, chat E2E, auth, GraphRAG.

---

## 13. Потенциально важные детали для будущего AI-агента

### Читать в первую очередь

0. [`AI_GUIDE.md`](AI_GUIDE.md) — маршруты по задачам
1. `flowsettings.py` — поведение по умолчанию
2. `app.py` + `src/ktem/main.py` + `src/ktem/app.py`
3. `src/ktem/pages/chat/__init__.py` — логика чата
4. `src/ktem/reasoning/simple.py` — RAG pipeline
5. `src/ktem/index/file/pipelines.py` + `index.py`

### Обычно не нужно

- `docs/theme/assets/*` (сгенерённые JS для MkDocs)
- `ktem_app_data/**` (runtime, большие бинарники)
- `uv.lock` целиком
- Все loader-утилиты, пока задача не про конкретный формат файла

### Где искать

| Задача | Путь |
|--------|------|
| Бизнес-логика QA | `src/ktem/reasoning/`, `src/kotaemon/indices/qa/` |
| UI | `src/ktem/pages/`, `src/ktem/assets/` |
| БД | `src/ktem/db/`, `src/ktem/index/file/index.py` (dynamic tables) |
| API провайдеров | `src/kotaemon/llms/`, `embeddings/` |
| Индексация | `src/ktem/index/file/pipelines.py`, `src/kotaemon/indices/ingests/` |
| Конфиг | `flowsettings.py`, `.env` |

### Хрупкие / сложные места

- **Динамические SQL-таблицы** per index id — миграции вручную (`scripts/migrate/`).
- **GraphRAG / LightRAG** — тяжёлые зависимости, флаги в flowsettings, отдельные index types.
- **theflow deserialize** — ошибки в `__type__` проявляются только в runtime.
- **Gradio + большие файлы** — таймауты, `GRADIO_TEMP_DIR`, rate limits (`utils/rate_limit.py`).
- **Расхождение upstream paths** (`libs/kotaemon` vs `src/`) — ломает скрипты/CI/docs.

### Conventions

- Компоненты наследуют `kotaemon.base.BaseComponent`, поля через `Node`/`Param`.
- Конфиг классов — dict с `"__type__": "dotted.path.Class"`.
- Именование: `KH_*` в settings, pluggy hook `ktem_declare_extensions`.
- `safe=False` при import_dotted_string в доверенных путях из settings.

---

## 14. Карта зависимостей

```
app.py
  └─ ktem.main.App
       └─ BaseApp (ktem.app)
            ├─ IndexManager → FileIndex / Graph*Index
            │     └─ pipelines (index/retrieval)
            │           └─ kotaemon: loaders → split → embed → storages
            ├─ reasonings registry ← flowsettings.KH_REASONINGS
            │     └─ e.g. FullQAPipeline
            │           └─ llms.manager → ChatLLM
            │           └─ retrievers → VectorRetrieval → Chroma/LanceDB
            │           └─ AnswerWithContextPipeline (citation_qa)
            └─ pages/chat/ChatPage
                  └─ Conversation (SQLModel) ↔ user messages

flowsettings.py + .env
  └─ KH_LLMS / KH_EMBEDDINGS → *Manager → SQLite *Table
  └─ KH_DOCSTORE / KH_VECTORSTORE → components.get_*()
```

SSO-вариант: `uvicorn sso_app:app` → FastAPI → Gradio mount → тот же `App`.

---

## 15. Что непонятно / требует внимания

| Вопрос | Почему | Что проверить |
|--------|--------|---------------|
| CI `libs/kotaemon` + Python 3.10 | Несовместимо с layout форка и `requires-python >=3.11` | `.github/workflows/unit-test.yaml`, правка matrix и `pytest tests` |
| `.omx/` | Служебные JSON (notify/update) | Игнорировать для RAG-логики |
| `Makefile` `dev --reload` | `app.py` не парсит argparse для reload | Работает ли `make dev` |
| Версия пакета в CI vs `pyproject` 0.1.0 | setuptools-git-versioning в workflow | `pyproject.toml` / tags |
| Полный набор тестов upstream | В форке один тест | Репозиторий Cinnamon/kotaemon |
| Двойной смысл `LOCAL_MODEL` | Ollama model name в `.env.example` vs path в `serve_local.py` | Какой сценарий использует пользователь |

---

## 16. Быстрый старт для Codex

Сначала [`AI_GUIDE.md`](AI_GUIDE.md), затем:

1. Открой `flowsettings.py` — пойми индексы, stores, reasoning и feature flags.
2. Прочитай `app.py` → `src/ktem/main.py` — карта вкладок UI.
3. Для изменений чата: `src/ktem/pages/chat/__init__.py` + `src/ktem/reasoning/simple.py`.
4. Для загрузки/поиска по документам: `src/ktem/index/file/pipelines.py` и `index.py`.
5. Для нового LLM-провайдера: `src/kotaemon/llms/` + запись в `flowsettings.KH_LLMS` + UI `src/ktem/llms/`.
6. Запуск: `pip install -r requirements_gerageragera39.txt && pip install -e . && python app.py` (порт 7860).
7. Данные пользователя: `ktem_app_data/user_data/` — не трогать без бэкапа.
8. Не ищи REST routes для чата — только Gradio.
9. Избегай правок `uv.lock` и путей `libs/kotaemon` без синхронизации структуры репо.
10. Секреты только через `.env`; в ответах и коммитах не дублировать ключи.
