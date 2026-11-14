#!/usr/bin/env bash
# Cria o ambiente do Unlimited-OCR no WSL2 e baixa os modelos (OCR e resumo).
# Uso (dentro do WSL):  bash scripts/setup_wsl.sh
set -euo pipefail

VENV="${UOCR_VENV:-$HOME/.venvs/uocr}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v nvidia-smi >/dev/null; then
  echo "ERRO: nvidia-smi não encontrado no WSL. Atualize o driver NVIDIA do Windows." >&2
  exit 1
fi

if ! command -v gcc >/dev/null; then
  # Os kernels Triton do modelo são compilados em tempo de execução e precisam de gcc.
  echo ">> Instalando gcc (necessário para o Triton)"
  if [ "$(id -u)" = 0 ]; then apt-get update -qq && apt-get install -y -qq gcc libc6-dev
  else sudo apt-get update -qq && sudo apt-get install -y -qq gcc libc6-dev; fi
fi

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo ">> Instalando uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo ">> Instalando dependências do pyproject.toml em $VENV (Python 3.10)"
export UV_PROJECT_ENVIRONMENT="$VENV"
cd "$ROOT"
if [ -f uv.lock ]; then uv sync --locked --python 3.10   # versões exatas do lock
else uv sync --python 3.10; fi                           # primeira vez: gera o uv.lock

echo ">> Checando CUDA"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponível'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

# Os dois modelos ficam no cache: depois disso tudo funciona sem internet.
echo ">> Baixando os modelos: OCR (~6,7 GB) e resumo Qwen2.5-7B (~15 GB)"
"$VENV/bin/python" -m uocr.core.hub

echo ">> Pronto."
