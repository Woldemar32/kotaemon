# Lite version (Python 3.11, dependencies from requirements_gerageragera39.txt)
FROM python:3.11-slim AS lite

# Common dependencies
RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        ssh \
        git \
        gcc \
        g++ \
        poppler-utils \
        libpoppler-dev \
        unzip \
        curl \
        cargo \
        && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup args
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG REQUIREMENTS_FILE=requirements_gerageragera39.txt

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV TARGETARCH=${TARGETARCH}

WORKDIR /app

# Download pdfjs (script only; keeps this layer cacheable)
COPY scripts/download_pdfjs.sh /app/scripts/download_pdfjs.sh
RUN sed -i 's/\r$//' /app/scripts/download_pdfjs.sh && chmod +x /app/scripts/download_pdfjs.sh
ENV PDFJS_PREBUILT_DIR="/app/libs/ktem/ktem/assets/prebuilt/pdfjs-dist"
RUN bash scripts/download_pdfjs.sh "$PDFJS_PREBUILT_DIR"

# Install pinned dependencies before copying the full tree (better layer cache)
COPY ${REQUIREMENTS_FILE} /app/${REQUIREMENTS_FILE}
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && .venv/bin/pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

# Application source and config
COPY . /app
COPY launch.sh /app/launch.sh
COPY .env.example /app/.env

RUN sed -i 's/\r$//' /app/launch.sh

# Optional: Microsoft GraphRAG (amd64 only)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        .venv/bin/pip install --no-cache-dir "graphrag<=0.3.6" future; \
    fi

ENTRYPOINT ["sh", "/app/launch.sh"]

# Full version (OCR, LibreOffice, PyTorch for unstructured)
FROM lite AS full

RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-jpn \
        libsm6 \
        libxext6 \
        libreoffice \
        ffmpeg \
        libmagic-dev \
        && \
    apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch is not pinned in requirements_gerageragera39.txt; install for unstructured
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        .venv/bin/pip install --no-cache-dir \
            torch torchvision torchaudio \
            --index-url "${TORCH_INDEX_URL}" \
            --extra-index-url https://pypi.org/simple; \
    else \
        .venv/bin/pip install --no-cache-dir torch torchvision torchaudio; \
    fi

ENV USE_LIGHTRAG=true

RUN /app/.venv/bin/python -c "from llama_index.core.readers.base import BaseReader"

ENTRYPOINT ["sh", "/app/launch.sh"]

# Ollama-bundled version
FROM full AS ollama

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN nohup bash -c "ollama serve &" && sleep 4 && ollama pull nomic-embed-text

ENV OLLAMA_ENABLED=true
ENTRYPOINT ["sh", "/app/launch.sh"]
