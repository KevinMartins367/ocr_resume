"""Base dos serviços: acesso ao modelo e à GPU, igual para o OCR e para o resumo.

Um serviço recebe um *provider* (função sem argumentos que devolve o modelo pronto, ver
`adapters.base.GpuSlot.provider`) em vez do modelo. Assim as entradas são validadas antes de
carregar qualquer coisa, e o mesmo serviço funciona na CLI (um modelo só) e na web (troca).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# Uma tarefa por vez na GPU, para todos os serviços: a troca de modelo (GpuSlot) e a inferência
# acontecem dentro dele. RLock: um serviço pode chamar outro sem travar.
GPU_LOCK = threading.RLock()


class ModelService:
    def __init__(self, provider: Callable[[], Any]):
        self.provider = provider

    @contextmanager
    def using_model(self) -> Iterator[Any]:
        """Reserva a GPU, carrega o modelo se preciso e o entrega para a tarefa."""
        with GPU_LOCK:
            yield self.provider()
