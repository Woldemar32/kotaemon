#!/usr/bin/env bash
# Start Kotaemon with Docker Compose (persistent ./ktem_app_data)
# Usage:
#   ./scripts/docker-up.sh
#   ./scripts/docker-up.sh --ollama
#   ./scripts/docker-up.sh --build

set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA=false
RERANKER=false
BUILD=false
for arg in "$@"; do
  case "$arg" in
    --ollama) OLLAMA=true ;;
    --reranker) RERANKER=true ;;
    --build) BUILD=true ;;
  esac
done

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example ..."
  cp .env.example .env
  echo "Edit .env then re-run."
fi

cmd=(docker compose)
if $OLLAMA; then cmd+=(--profile ollama); fi
if $RERANKER; then cmd+=(--profile reranker); fi
cmd+=(up -d)
if $BUILD; then cmd+=(--build); fi

echo "${cmd[*]}"
"${cmd[@]}"

echo ""
echo "App:  http://localhost:7860"
echo "Data: $(pwd)/ktem_app_data"
echo "Logs: docker compose logs -f kotaemon"
echo "Stop: docker compose down   (data kept unless you use -v)"
