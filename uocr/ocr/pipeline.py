"""Pipeline: páginas -> Unlimited-OCR-2 (em janelas de N páginas) -> Markdown + imagens.

O modelo não gera Markdown: gera blocos `<PAGE><BLOCK>label<LEVELn>[x1,y1,x2,y2]<FORK>texto`
(coordenadas em 0-1000). A conversão usa o pós-processamento oficial (`_postprocess.convert`),
aplicado ao documento inteiro, o que preserva a junção de tabelas/parágrafos entre páginas.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageDraw, ImageFont

from . import _postprocess as pp

MODES = {"auto": None, "gundam": True, "base": False}  # crop_mode do modelo
_COLORS = {"title": (220, 40, 40), "table": (30, 150, 60), "image": (200, 120, 0)}


@dataclass
class OCRResult:
    out_dir: Path
    markdown: str
    raw: str
    pages: int
    seconds: float
    tokens: int
    box_images: list[Path] = field(default_factory=list)


def _px(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (pp._scale(v) for v in bbox)
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def _save_layout(raw: str, images: list[Image.Image], out_dir: Path) -> list[Path]:
    """Desenha os blocos de layout por página e recorta as figuras com o mesmo nome
    que `convert()` usa no Markdown (pageN_x1_y1_x2_y2.jpg)."""
    font = ImageFont.load_default()
    canvases = [img.copy() for img in images]
    draws = [ImageDraw.Draw(c) for c in canvases]
    img_dir = out_dir / "images"
    for b in pp._parse_blocks(raw, page_base=1):
        page = b["page"] - 1
        if b["bbox"] is None or not (0 <= page < len(images)):
            continue
        w, h = images[page].size
        box = _px(b["bbox"], w, h)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        base = pp._base_type(b["label"])
        kind = "image" if base in pp.IMAGE_LABELS else base
        color = _COLORS.get(kind, (40, 90, 220))
        draws[page].rectangle(box, outline=color, width=4 if kind == "title" else 2)
        draws[page].text((box[0], max(0, box[1] - 12)), b["label"], fill=color, font=font)
        if kind == "image":
            img_dir.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = (pp._scale(v) for v in b["bbox"])
            images[page].crop(box).save(img_dir / f"page{b['page']}_{x1}_{y1}_{x2}_{y2}.jpg")
    paths = []
    for i, c in enumerate(canvases, 1):
        p = out_dir / "boxes" / f"page_{i}.jpg"
        c.save(p, quality=85)
        paths.append(p)
    return paths


def run_ocr(images: list[Image.Image], out_dir: str | Path, model, tokenizer, *,
            batch_pages: int = 4, mode: str = "auto", tokens_per_page: int = 8000,
            ngram_size: int = 35, stream: bool = False,
            progress: Callable[[int, int, str], None] | None = None) -> OCRResult:
    """Processa as páginas em janelas de `batch_pages` (limita VRAM) e grava em `out_dir`:
    result.md, result_raw.md, images/ e boxes/page_N.jpg.
    """
    out_dir = Path(out_dir)
    (out_dir / "boxes").mkdir(parents=True, exist_ok=True)
    crop_mode = MODES[mode]
    batch_pages = max(1, batch_pages)
    windows = [images[i:i + batch_pages] for i in range(0, len(images), batch_pages)]

    t0 = time.time()
    raw_parts, n_tokens, done = [], 0, 0
    for wi, win in enumerate(windows):
        if progress:
            progress(wi, len(windows), f"janela {wi + 1}/{len(windows)} "
                                       f"(páginas {done + 1}-{done + len(win)})")
        # O código remoto do modelo imprime debug ([DASH], [debug ...]) no stdout.
        quiet = contextlib.nullcontext() if stream else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            res = model.infer(
                tokenizer, image_file=win, crop_mode=crop_mode, eval_mode=not stream,
                max_new_tokens=tokens_per_page * len(win), no_repeat_ngram_size=ngram_size,
                ngram_window=1024 if len(win) > 1 else 128, return_dict=True)
        text = res["text"]
        if not text.lstrip().startswith("<PAGE>"):
            text = "<PAGE>" + text
        n_out = res.get("num_pages_out", text.count("<PAGE>"))
        if n_out != len(win):
            print(f"[aviso] janela {wi + 1}: {len(win)} página(s) enviadas, {n_out} retornadas "
                  f"({res.get('finish_reason')})", file=sys.stderr)
        raw_parts.append(text)
        n_tokens += len(res["token_ids"])
        done += len(win)
        torch.cuda.empty_cache()

    raw = "".join(raw_parts)
    markdown = pp.convert(raw).replace("](page", "](images/page") + "\n"
    (out_dir / "result.md").write_text(markdown, encoding="utf-8")
    (out_dir / "result_raw.md").write_text(raw, encoding="utf-8")
    box_paths = _save_layout(raw, images, out_dir)
    if progress:
        progress(len(windows), len(windows), "concluído")
    return OCRResult(out_dir=out_dir, markdown=markdown, raw=raw, pages=len(images),
                     seconds=time.time() - t0, tokens=n_tokens, box_images=box_paths)
