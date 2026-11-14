#!/usr/bin/env bash
# Executa um comando do projeto no ambiente uv do WSL (equivalente ao "npm run").
# Ex.: bash scripts/run.sh uocr-web --port 7860
#      bash scripts/run.sh uocr 'D:\docs\arquivo.pdf'   (caminhos do Windows viram /mnt/d/...)
# O venv fica em ~/.venvs/uocr (disco do WSL): em /mnt/d o torch ficaria lento.
export UV_PROJECT_ENVIRONMENT="${UOCR_VENV:-$HOME/.venvs/uocr}"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Converte caminhos do Windows (C:\x ou C:/x) para o WSL. Relativos ficam como estão:
# o comando roda na raiz do projeto, igual ao make.
args=()
for a in "$@"; do
    if [[ $a =~ ^[A-Za-z]:[\\/] ]] && command -v wslpath >/dev/null; then
        a="$(wslpath -a "$a")"
    fi
    args+=("$a")
done

exec uv run --frozen "${args[@]}"
