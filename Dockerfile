# syntax=docker/dockerfile:1
# Unlimited-OCR (Linux + GPU NVIDIA). Estágios: base -> test (pytest) | app (interface web).
FROM python:3.10-slim-bookworm AS base

# Marca todas as imagens do projeto (app, test, dev). Ao reconstruir, a versão antiga fica
# como "<none>"; `make docker-clean` (e os alvos docker-*) apagam só as que têm este rótulo.
LABEL com.unlimited-ocr.project="uocr"

# gcc: o Triton compila os kernels do modelo em tempo de execução.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/hf \
    TRITON_CACHE_DIR=/data/triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependências (camada em cache enquanto o uv.lock não mudar).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) Código do projeto.
COPY README.md ./
COPY uocr ./uocr
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# Testes com mocks: não precisam de GPU nem do modelo.
FROM base AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable
COPY tests ./tests
CMD ["pytest"]


# Dev Container do VS Code (.devcontainer/): testes + git/make para editar dentro do container.
# procps: pkill, usado pelo `make stop`.
FROM test AS dev
RUN apt-get update \
 && apt-get install -y --no-install-recommends git make procps \
 && rm -rf /var/lib/apt/lists/*


# Imagem final (padrão): interface web. O modelo é baixado no 1º uso para o volume /data.
FROM base AS app
EXPOSE 7860
CMD ["uocr-web", "--host", "0.0.0.0", "--port", "7860", "--output", "/app/output/web"]
