# AI Guide for this repository

Краткая инструкция для Codex, Cursor и других AI-агентов. Полный контекст: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — **читайте его первым**, затем только файлы по задаче.

---

## Что это

**Kotaemon fork** — RAG-приложение для чата с документами (PDF, Office, изображения и др.). UI на **Gradio** (`ktem`), переиспользуемая библиотека RAG — **`kotaemon`** в `src/`. Нет отдельного React-frontend; основной API — Gradio callbacks, не REST.

Исходники: `src/kotaemon/`, `src/ktem/` (не `libs/kotaemon` из upstream). Python **≥ 3.11** (`pyproject.toml`).

---

## Самые важные файлы

| Задача | Сначала читать |
|--------|----------------|
| Запуск приложения | `app.py`, `launch.sh`, `Makefile` |
| Конфиг моделей и индексов | `flowsettings.py`, `.env.example`, `config_example.txt` |
| Чат (UI + события) | `src/ktem/pages/chat/__init__.py`, `src/ktem/main.py` |
| Reasoning / RAG pipeline | `src/ktem/reasoning/simple.py`, `src/kotaemon/indices/qa/citation_qa.py` |
| Индексация файлов | `src/ktem/index/file/pipelines.py`, `src/ktem/index/file/index.py` |
| LLM providers | `src/kotaemon/llms/`, `src/ktem/llms/manager.py`, `flowsettings.KH_LLMS` |
| Embeddings | `src/kotaemon/embeddings/`, `src/ktem/embeddings/manager.py` |
| Database | `src/ktem/db/models.py`, `src/ktem/db/engine.py`, `src/ktem/index/models.py` |
| Docker | `Dockerfile`, `launch.sh`, `README.md` |
| Tests / CI | `tests/`, `.github/workflows/unit-test.yaml`, `pyproject.toml` |
| Оболочка приложения | `src/ktem/app.py`, `src/ktem/components.py` |
| Локальная `.gguf` (llama.cpp) | `scripts/serve_local.py`, `scripts/server_llamacpp_*.bat/sh` |
| GraphRAG / LightRAG | `flowsettings.py` (флаги `USE_*`), `src/ktem/index/file/graph/` |

---

## Что обычно не читать

Игнорируйте, если задача **прямо** не про них:

| Путь | Причина |
|------|---------|
| `ktem_app_data/` | Runtime: SQLite, uploads, vectorstore, HF cache — не исходники |
| `uv.lock` | Большой lock; может ссылаться на legacy `libs/kotaemon` |
| `__pycache__/`, `.venv/` | Кэш / окружение |
| `docs/theme/assets/` | Сгенерированные assets MkDocs |
| `.omx/` | Служебные state-файлы |
| `dataset/` | Тестовые документы, не код |
| `templates/project-default/` | Cookiecutter шаблон нового проекта |
| Adobe / Azure DI / GraphRAG MS | Тяжёлые опции — только при задаче на эту интеграцию |
| Весь `src/kotaemon/loaders/` | Только если меняется парсинг конкретного формата |

---

## Частые задачи и куда идти

| Задача | Куда |
|--------|------|
| Изменить поведение чата | `src/ktem/pages/chat/`, `src/ktem/reasoning/` |
| Изменить промпты / формат ответа | `src/kotaemon/indices/qa/`, `src/kotaemon/llms/prompts/`, `src/ktem/reasoning/prompt_optimization/` |
| Изменить индексацию документов | `src/ktem/index/file/pipelines.py`, `src/kotaemon/indices/ingests/` |
| Добавить Ollama / local model | `.env`, `flowsettings.py` (`LOCAL_MODEL`, `KH_OLLAMA_URL`), UI `src/ktem/llms/`, шпаргалка `config_example.txt` |
| Запустить llama.cpp для `.gguf` | `scripts/serve_local.py` (`LOCAL_MODEL` = **путь к файлу**), затем LLM в Resources с `base_url` на порт 31415 |
| Добавить новый LLM provider | `src/kotaemon/llms/`, запись в `KH_LLMS`, `src/ktem/llms/ui.py` + `db.py` |
| Исправить Docker | `Dockerfile`, `launch.sh`, `README.md` |
| Исправить тесты / CI | `tests/`, `.github/workflows/unit-test.yaml` — **не** `cd libs/kotaemon`; использовать `pytest tests`, Python 3.11+ |
| GraphRAG / LightRAG | `flowsettings.GRAPHRAG_INDEX_TYPES`, `src/ktem/index/file/graph/`, `settings.yaml.example` |

---

## Команды

```bash
# Install (из корня)
make install
# или: python -m venv .venv && pip install -r requirements_gerageragera39.txt && pip install -e .

# Run (Gradio :7860)
python app.py

# Локальный llama.cpp для .gguf (после пути в .env)
python scripts/serve_local.py

# Docker
docker build --target full -t kotaemon:full .
docker run --rm -p 7860:7860 -v "%cd%\ktem_app_data:/app/ktem_app_data" --env-file .env kotaemon:full

# Tests
pytest tests

# Lint (если настроен pre-commit)
pre-commit run --all-files
```

---

## Важные предупреждения

1. **Не коммитьте и не выводите** содержимое `.env` — только имена переменных из `.env.example`.
2. **Не удаляйте и не правьте** `ktem_app_data/` без бэкапа — там БД и индексы пользователя.
3. **Не доверяйте путям `libs/kotaemon`** в CI, `uv.lock`, `run_*.sh`, `mkdocs.yml` — в форке код в `src/` (см. §2 в `PROJECT_CONTEXT.md`).
4. **`LOCAL_MODEL`**: в `flowsettings`/Ollama — имя модели; в `serve_local.py` — путь к `.gguf`. Не путать сценарии.
5. **`config_example.txt`** не подхватывается приложением — это пример полей для вкладки Resources.
6. Перед правками: **`PROJECT_CONTEXT.md` → 2–5 релевантных файлов** — не сканировать весь репозиторий.
