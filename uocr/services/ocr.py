"""Serviço de OCR usado pela CLI e pela web: arquivos -> documento, .zip -> lote."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.util import Progress
from ..ocr.batch import BatchResult, run_zip
from ..ocr.pdf import load_inputs
from ..ocr.pipeline import OCRResult, run_ocr
from .base import ModelService


@dataclass
class OcrOptions:
    dpi: int = 200
    pages: str | None = None
    batch_pages: int = 4
    mode: str = "auto"
    tokens_per_page: int = 8000
    stream: bool = False

    def pipeline_kw(self) -> dict:
        return dict(batch_pages=self.batch_pages, mode=self.mode,
                    tokens_per_page=self.tokens_per_page, stream=self.stream)


class OcrService(ModelService):
    """O provider devolve `(model, tokenizer)` (ver `adapters.ocr.OcrAdapter`)."""

    def document(self, paths: list[str | Path], out_dir: str | Path, opts: OcrOptions = OcrOptions(),
                 progress: Progress | None = None) -> OCRResult:
        """PDFs/imagens -> um documento em `out_dir` (result.md, images/, boxes/)."""
        images = load_inputs(paths, dpi=opts.dpi, pages=opts.pages)   # valida antes de carregar
        with self.using_model() as (model, tok):
            return run_ocr(images, out_dir, model, tok, progress=progress, **opts.pipeline_kw())

    def batch(self, zip_path: str | Path, out_dir: str | Path, opts: OcrOptions = OcrOptions(),
              progress: Progress | None = None) -> BatchResult:
        """.zip -> um .md por arquivo em `out_dir`, espelhando as pastas."""
        with self.using_model() as (model, tok):
            return run_zip(zip_path, out_dir, model, tok, dpi=opts.dpi, pages=opts.pages,
                           progress=progress, **opts.pipeline_kw())
