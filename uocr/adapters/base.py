"""Adapters de modelo: o que é igual em todo modelo do Hugging Face numa GPU pequena.

`ModelAdapter` cuida de validar a quantização, exigir CUDA, resolver o modelo para a pasta
local (offline), manter o modelo carregado e liberá-lo. Cada modelo só implementa `_build`.
`GpuSlot` garante um modelo por vez na GPU (OCR e resumo não cabem juntos em 6 GB).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..core import hub
from ..core.gpu import free, vram_report


def bnb_config(quant: str, skip_modules: list[str] | None = None):
    """BitsAndBytesConfig para `4bit` (NF4, double quant, cálculo em bf16) ou `8bit`;
    None para os modos sem quantização. `skip_modules` ficam em bf16 (None = padrão do
    transformers, que preserva o lm_head; [] quantiza tudo)."""
    from transformers import BitsAndBytesConfig

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip_modules)
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)
    return None


class ModelAdapter(ABC):
    """Um modelo do Hugging Face carregado sob demanda. `load()` devolve o mesmo objeto até
    `unload()`. Subclasses definem `kind`, `quant_choices` e `_build(path)`."""

    kind: str = "modelo"                 # nome exibido nas mensagens
    quant_choices: tuple[str, ...] = ()

    def __init__(self, model_id: str, quant: str):
        if quant not in self.quant_choices:
            raise ValueError(f"quant inválido: {quant!r} (use {', '.join(self.quant_choices)})")
        self.model_id, self.quant = model_id, quant
        self._loaded: Any = None

    def __repr__(self) -> str:
        return f"{self.kind} {self.model_id} ({self.quant})"

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> Any:
        if self._loaded is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA indisponível. Rode dentro do WSL2 com driver NVIDIA.")
            self._loaded = self._build(hub.local_model_path(self.model_id))
        return self._loaded

    def unload(self) -> None:
        if self._loaded is not None:
            self._loaded = None
            free()

    @abstractmethod
    def _build(self, path: str) -> Any:
        """Carrega a partir da pasta local `path` e devolve o que os serviços usam."""


class GpuSlot:
    """Uma vaga na GPU para vários adapters: pedir um descarrega os outros."""

    def __init__(self, *adapters: ModelAdapter, log=print):
        self.adapters = adapters
        self.log = log

    def get(self, adapter: ModelAdapter) -> Any:
        if not adapter.loaded:
            for other in self.adapters:
                if other is not adapter:
                    other.unload()
            self.log(f">> Carregando {adapter}...")
            adapter.load()
            self.log(f">> {vram_report()}")
        return adapter.load()

    def provider(self, adapter: ModelAdapter):
        """Função sem argumentos que devolve o modelo pronto (o que os serviços recebem)."""
        return lambda: self.get(adapter)
