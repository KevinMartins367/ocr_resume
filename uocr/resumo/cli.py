"""CLI: uocr-resumo output/lote [outra/pasta ...]"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.resumo import DEFAULT_SUMMARY_MODEL, SUMMARY_QUANT_CHOICES
from .text import DEFAULT_MAX_INPUT, DEFAULT_MAX_NEW


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uocr-resumo",
        description="Grava um _resumo.md em cada pasta com .md (ex.: saída de um lote .zip). "
                    "Sem resumo geral: cada pasta resume só os próprios arquivos.")
    p.add_argument("folders", nargs="+", help="pastas geradas pelo OCR (ex.: output/lote)")
    p.add_argument("--model", default=DEFAULT_SUMMARY_MODEL, help="id do modelo no Hugging Face")
    p.add_argument("--quant", choices=SUMMARY_QUANT_CHOICES, default="4bit",
                   help="4bit cabe em 6 GB; 8bit/bf16 pedem GPU maior")
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT,
                   help=f"tokens de entrada por chamada; pastas maiores são resumidas em partes "
                        f"(padrão: {DEFAULT_MAX_INPUT})")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW,
                   help=f"tamanho máximo de cada resumo (padrão: {DEFAULT_MAX_NEW})")
    p.add_argument("--instrucao", default="", help='pedido extra, ex.: "foque em datas e valores"')
    p.add_argument("--force", action="store_true", help="refaz também as pastas já resumidas")
    p.add_argument("--stream", action="store_true", help="mostra o texto enquanto é gerado")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for f in args.folders:
        if not Path(f).is_dir():
            raise SystemExit(f"pasta não encontrada: {f}")

    from ..adapters.base import GpuSlot
    from ..adapters.resumo import QwenAdapter
    from ..core.util import print_progress
    from ..services.resumo import SummaryOptions, SummaryService

    adapter = QwenAdapter(args.model, args.quant, stream=args.stream)
    service = SummaryService(GpuSlot(adapter).provider(adapter), args.model)   # carrega no 1º uso
    opts = SummaryOptions(max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                          instruction=args.instrucao, force=args.force)
    failed = False
    for f in args.folders:
        try:
            res = service.summarize(f, opts, print_progress)
        except ValueError as e:
            print(f">> {e}")
            failed = True
            continue
        for folder, err in res.errors:
            print(f"   [erro] {folder}: {err}")
        skipped = f", {len(res.up_to_date)} já resumida(s)" if res.up_to_date else ""
        print(f">> {res.root}: {len(res.done)} pasta(s) resumida(s){skipped}, "
              f"{len(res.errors)} erro(s), {res.seconds:.0f}s")
        failed |= bool(res.errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
