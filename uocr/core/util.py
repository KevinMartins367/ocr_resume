"""Pequenas utilidades compartilhadas."""
from __future__ import annotations

import re
from typing import Callable

# Callback de progresso usado pelos pipelines: (passo atual, total, mensagem).
Progress = Callable[[int, int, str], None]


def natural_key(s: str):
    """Ordem natural: `pag2` antes de `pag10`."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def print_progress(_i: int, _n: int, msg: str) -> None:
    """Progresso no terminal (CLIs)."""
    print(f"   {msg}", flush=True)
