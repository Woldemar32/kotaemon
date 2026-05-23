.PHONY: install install-gpu run dev docker-build docker-up docker-up-ollama docker-down docker-logs docker-ps

install:
	uv venv .venv --python 3.11
	uv pip install -r requirements_gerageragera39.txt
	uv pip install -e ".[dev]"

install-gpu: install
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

run:
	.venv/Scripts/python app.py
	# .venv/bin/python app.py      # Linux/Mac

dev:
	.venv/Scripts/python app.py --reload

# --- Docker Compose (persistent data in ./ktem_app_data) ---
docker-build:
	docker compose build

docker-up:
	docker compose up -d --build

docker-up-ollama:
	docker compose --profile ollama up -d --build

docker-up-full:
	docker compose --profile ollama --profile reranker up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f kotaemon

docker-ps:
	docker compose ps
