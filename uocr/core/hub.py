"""Modelos do Hugging Face sem depender da internet.

Tudo roda offline depois que os modelos são baixados uma vez (`make baixar`). Antes de
carregar, o id (`baidu/Unlimited-OCR-2`) vira a pasta local do cache: o transformers carrega
direto do disco, sem consultar o Hub. Isso evita esperar timeouts quando não há rede.

    python -m uocr.core.hub              # baixa o modelo de OCR e o de resumo
    python -m uocr.core.hub --so-ocr     # só o de OCR
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Arquivos que o carregamento não usa (o repositório do OCR traz PDFs, imagens e wheels).
IGNORE = ["*.pdf", "assets/*", "wheel/*", "*.pth", "*.bin", "*.gguf", "original/*"]


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


def local_model_path(model_id: str) -> str:
    """Pasta local do modelo. Usa o cache sem tocar na rede. Se não houver cache e houver
    internet, baixa; offline, explica como baixar."""
    if Path(model_id).is_dir():
        return model_id
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        pass
    if _offline():
        raise RuntimeError(f"o modelo {model_id} não está baixado e o modo offline está ativo. "
                           "Com internet, rode uma vez: make baixar")
    try:
        print(f">> {model_id} não está no cache: baixando (só desta vez)...", flush=True)
        return snapshot_download(model_id, ignore_patterns=IGNORE)
    except Exception as e:
        raise RuntimeError(f"o modelo {model_id} não está baixado e não foi possível baixar "
                           f"({type(e).__name__}). Com internet, rode uma vez: make baixar") from e


def download(model_ids: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for mid in model_ids:
        print(f">> {mid}", flush=True)
        print(f"   {snapshot_download(mid, ignore_patterns=IGNORE)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    from ..adapters.ocr import DEFAULT_MODEL
    from ..adapters.resumo import DEFAULT_SUMMARY_MODEL

    p = argparse.ArgumentParser(prog="python -m uocr.core.hub",
                                description="Baixa os modelos para uso offline.")
    p.add_argument("--so-ocr", action="store_true", help=f"só o modelo de OCR ({DEFAULT_MODEL})")
    a = p.parse_args(argv)
    models = [DEFAULT_MODEL] if a.so_ocr else [DEFAULT_MODEL, DEFAULT_SUMMARY_MODEL]
    download(models)
    print(">> Pronto: os modelos funcionam sem internet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
