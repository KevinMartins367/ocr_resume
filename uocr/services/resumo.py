"""Serviço de resumo usado pela CLI e pela web: um `_resumo.md` por pasta."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.util import Progress
from ..resumo.folders import SummaryResult, find_folders, summarize_tree
from ..resumo.text import DEFAULT_MAX_INPUT, DEFAULT_MAX_NEW
from .base import ModelService


@dataclass
class SummaryOptions:
    max_input_tokens: int = DEFAULT_MAX_INPUT
    max_new_tokens: int = DEFAULT_MAX_NEW
    instruction: str = ""
    force: bool = False


class SummaryService(ModelService):
    """O provider devolve um `Summarizer` (ver `adapters.resumo.QwenAdapter`)."""

    def __init__(self, provider: Callable[[], Any], model_id: str):
        super().__init__(provider)
        self.model_id = model_id

    def summarize(self, folder: str | Path, opts: SummaryOptions = SummaryOptions(),
                  progress: Progress | None = None) -> SummaryResult:
        folder = Path(folder)
        if not folder.is_dir():                     # valida antes de carregar o modelo
            raise ValueError(f"pasta não encontrada: {folder}")
        if not find_folders(folder):
            raise ValueError(f"nenhum .md em {folder}")
        with self.using_model() as llm:
            return summarize_tree(folder, llm, model_id=self.model_id, progress=progress,
                                  **asdict(opts))
