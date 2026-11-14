"""Conversão de entradas (PDF/imagens) em lista de páginas PIL."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-3,5' -> [0, 1, 2, 4] (índices 0-based, dentro de [0, total))."""
    if not spec:
        return list(range(total))
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a or 1), int(b or total)
        else:
            start = end = int(part)
        out.extend(i - 1 for i in range(start, end + 1) if 1 <= i <= total)
    return sorted(set(out))


def pdf_to_images(pdf: str | Path | bytes, dpi: int = 200, pages: str | None = None) -> list[Image.Image]:
    """`pdf` pode ser um caminho ou o conteúdo do arquivo em bytes."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf, filetype="pdf") if isinstance(pdf, (bytes, bytearray)) else fitz.open(str(pdf))
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        imgs = []
        for i in parse_pages(pages, doc.page_count):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return imgs
    finally:
        doc.close()


def load_file(name: str, data: str | Path | bytes, dpi: int = 200,
              pages: str | None = None) -> list[Image.Image]:
    """Um arquivo (caminho ou bytes) -> páginas. PDFs são rasterizados; imagens recebem correção EXIF."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return pdf_to_images(data, dpi=dpi, pages=pages)
    if ext in IMAGE_EXTS:
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        return [ImageOps.exif_transpose(Image.open(src)).convert("RGB")]
    raise ValueError(f"formato não suportado: {Path(name).name}")


def load_inputs(paths: list[str | Path], dpi: int = 200, pages: str | None = None) -> list[Image.Image]:
    """PDFs viram páginas; imagens entram como estão. Ordem preservada."""
    images: list[Image.Image] = []
    for p in map(Path, paths):
        images.extend(load_file(p.name, p, dpi=dpi, pages=pages))
    if not images:
        raise ValueError("nenhuma página para processar")
    return images
