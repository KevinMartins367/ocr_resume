"""CLI: python -m uocr arquivo.pdf [img.png ...] -o output"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.ocr import DEFAULT_MODEL, DEFAULT_QUANT, QUANT_CHOICES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uocr", description="OCR de PDFs/imagens com Unlimited-OCR-2 → Markdown")
    p.add_argument("inputs", nargs="+",
                   help="PDFs, imagens e/ou .zip (o zip gera um .md por arquivo, na mesma estrutura de pastas)")
    p.add_argument("-o", "--output", default="output", help="pasta de saída (padrão: output)")
    p.add_argument("--quant", choices=QUANT_CHOICES, default=DEFAULT_QUANT,
                   help=f"modo de carga do modelo (padrão: {DEFAULT_QUANT})")
    p.add_argument("--offload-layers", type=int, default=6, help="camadas MoE na CPU no modo offload")
    p.add_argument("--model", default=DEFAULT_MODEL, help="id do modelo no Hugging Face")
    p.add_argument("--dpi", type=int, default=200, help="DPI ao rasterizar PDFs (padrão: 200)")
    p.add_argument("--pages", help="páginas do PDF, ex.: 1-3,5")
    p.add_argument("--batch-pages", type=int, default=4, help="páginas por chamada ao modelo (padrão: 4)")
    p.add_argument("--mode", choices=("auto", "gundam", "base"), default="auto",
                   help="auto = tiles p/ 1 página, visão global p/ várias")
    p.add_argument("--tokens-per-page", type=int, default=8000, help="limite de tokens gerados por página")
    p.add_argument("--combine", action="store_true",
                   help="trata as entradas (exceto .zip) como um único documento")
    p.add_argument("--stream", action="store_true", help="mostra os tokens enquanto gera")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for f in args.inputs:
        if not Path(f).is_file():
            raise SystemExit(f"arquivo não encontrado: {f}")

    from ..adapters.base import GpuSlot
    from ..adapters.ocr import OcrAdapter
    from ..core.gpu import vram_report
    from ..core.util import print_progress
    from .batch import REPORT_NAME
    from ..services.ocr import OcrOptions, OcrService

    adapter = OcrAdapter(args.model, args.quant, args.offload_layers)
    service = OcrService(GpuSlot(adapter).provider(adapter))   # carrega no 1º uso
    opts = OcrOptions(dpi=args.dpi, pages=args.pages, batch_pages=args.batch_pages, mode=args.mode,
                      tokens_per_page=args.tokens_per_page, stream=args.stream)
    out = Path(args.output)
    used: set[str] = set()

    def out_name(first: Path) -> str:
        name = first.stem if first.stem not in used else f"{first.stem}_{first.suffix.lstrip('.')}"
        used.add(name)
        return name

    # .zip: um .md por arquivo, espelhando as pastas do zip
    zips = [f for f in args.inputs if f.lower().endswith(".zip")]
    for z in zips:
        name = out_name(Path(z))
        print(f">> {name}.zip (lote)")
        res = service.batch(z, out / name, opts, print_progress)
        extra = f" | veja {REPORT_NAME}" if res.errors or res.skipped else ""
        print(f">> {res.out_dir}  ({len(res.done)} .md, {len(res.errors)} erro(s), "
              f"{len(res.skipped)} ignorado(s), {res.seconds:.0f}s){extra}")

    others = [f for f in args.inputs if f not in zips]
    docs = ([others] if args.combine else [[f] for f in others]) if others else []
    for files in docs:
        name = out_name(Path(files[0]))
        print(f">> {name}")
        res = service.document(files, out / name, opts, print_progress)
        print(f">> {res.out_dir / 'result.md'}  ({res.pages} pág., {res.tokens} tokens, "
              f"{res.seconds:.0f}s) | {vram_report()}")
    return 0
