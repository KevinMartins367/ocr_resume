"""Utilidades de GPU compartilhadas pelo OCR e pelo resumo."""
from __future__ import annotations

import gc

import torch


def free() -> None:
    """Devolve à GPU a memória de modelos descartados (depois de limpar os caches)."""
    gc.collect()
    torch.cuda.empty_cache()


def vram_report() -> str:
    if not torch.cuda.is_available():
        return "sem CUDA"
    gb = 1024 ** 3
    return (f"VRAM alocada {torch.cuda.memory_allocated() / gb:.2f} GB | "
            f"pico {torch.cuda.max_memory_allocated() / gb:.2f} GB | "
            f"total {torch.cuda.get_device_properties(0).total_memory / gb:.2f} GB")
