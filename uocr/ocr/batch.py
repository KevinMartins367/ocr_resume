"""Lote: .zip com pastas de imagens/PDFs -> um .md por arquivo, espelhando a estrutura.

    entrada.zip                     saída/
    ├─ cap1/pag1.jpg        ->      ├─ cap1/pag1.md
    ├─ cap1/pag2.jpg        ->      ├─ cap1/pag2.md  (+ pag2_images/ se houver figuras)
    └─ anexos/contrato.pdf  ->      └─ anexos/contrato.md

Os arquivos são lidos direto do zip (nada é extraído para o disco). Caminhos com `..`,
absolutos ou com drive são rejeitados. Um arquivo com erro não interrompe o lote.
"""
from __future__ import annotations

import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import torch

from ..core.util import Progress, natural_key
from .pdf import SUPPORTED_EXTS, load_file
from .pipeline import run_ocr

MAX_FILES = 10_000
MAX_MEMBER_BYTES = 512 * 1024 ** 2
_SKIP_NAMES = {"thumbs.db", "desktop.ini"}
REPORT_NAME = "_relatorio.txt"


@dataclass
class BatchResult:
    out_dir: Path
    done: list[str] = field(default_factory=list)                    # .md gerados (relativos)
    errors: list[tuple[str, str]] = field(default_factory=list)      # (arquivo, erro)
    skipped: list[str] = field(default_factory=list)                 # ignorados (formato etc.)
    seconds: float = 0.0


def _decode_name(info: zipfile.ZipInfo) -> str:
    """Zips do Windows sem a flag UTF-8 gravam nomes no code page OEM (cp850 em pt-BR)."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437")  # zipfile decodificou como cp437: volta aos bytes originais
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp850")


def _safe_rel(name: str) -> PurePosixPath | None:
    """Caminho relativo seguro dentro do zip, ou None (.., drive, vazio)."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        return None
    return PurePosixPath(*parts)


def list_entries(zf: zipfile.ZipFile) -> tuple[list[tuple[PurePosixPath, zipfile.ZipInfo]], list[str]]:
    """Arquivos suportados do zip (ordem natural) e a lista do que foi ignorado."""
    entries, skipped = [], []
    for info in zf.infolist():
        if info.is_dir() or info.filename.endswith(("/", "\\")):
            continue
        rel = _safe_rel(_decode_name(info))
        if rel is None:
            skipped.append(f"{info.filename} (caminho inválido)")
            continue
        if any(p.startswith(".") or p == "__MACOSX" for p in rel.parts) or rel.name.lower() in _SKIP_NAMES:
            continue  # lixo de sistema: ignora em silêncio
        if rel.suffix.lower() not in SUPPORTED_EXTS:
            skipped.append(f"{rel} (formato não suportado)")
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            skipped.append(f"{rel} (maior que {MAX_MEMBER_BYTES // 1024 ** 2} MB)")
            continue
        entries.append((rel, info))
    if len(entries) > MAX_FILES:
        raise ValueError(f"o .zip tem {len(entries)} arquivos (máximo {MAX_FILES})")
    entries.sort(key=lambda e: natural_key(str(e[0])))
    return entries, skipped


def _md_target(rel: PurePosixPath, used: set[str]) -> PurePosixPath:
    """foto.jpg -> foto.md; se já existir (foto.png na mesma pasta), foto_png.md."""
    target = rel.with_suffix(".md")
    if str(target).lower() in used:
        target = rel.with_name(f"{rel.stem}_{rel.suffix.lstrip('.').lower()}.md")
    used.add(str(target).lower())
    return target


def _place(result, out_dir: Path, md_rel: PurePosixPath) -> None:
    """Grava o .md no caminho espelhado e leva as figuras para <nome>_images/."""
    target = out_dir.joinpath(*md_rel.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    md = result.markdown
    src_imgs = result.out_dir / "images"
    if src_imgs.is_dir() and any(src_imgs.iterdir()):
        assets = f"{md_rel.stem}_images"
        shutil.copytree(src_imgs, target.parent / assets, dirs_exist_ok=True)
        md = md.replace("](images/", f"]({quote(assets)}/")
    target.write_text(md, encoding="utf-8")


def _write_report(res: BatchResult) -> None:
    if not (res.errors or res.skipped):
        return
    lines = [f"Gerados: {len(res.done)} | Erros: {len(res.errors)} | Ignorados: {len(res.skipped)}", ""]
    if res.errors:
        lines += ["ERROS:"] + [f"  {f}: {e}" for f, e in res.errors] + [""]
    if res.skipped:
        lines += ["IGNORADOS:"] + [f"  {s}" for s in res.skipped]
    (res.out_dir / REPORT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_zip(zip_path: str | Path, out_dir: str | Path, model, tokenizer, *, dpi: int = 200,
            pages: str | None = None, progress: Progress | None = None,
            **ocr_kw) -> BatchResult:
    """OCR de cada imagem/PDF do zip; grava os .md em `out_dir` com a mesma estrutura de pastas.
    `ocr_kw` vai para `run_ocr` (batch_pages, mode, tokens_per_page...)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = BatchResult(out_dir=out_dir)
    t0 = time.time()
    used: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        entries, res.skipped = list_entries(zf)
        if not entries:
            raise ValueError("o .zip não tem imagens nem PDFs")
        for i, (rel, info) in enumerate(entries):
            if progress:
                progress(i, len(entries), f"{i + 1}/{len(entries)}: {rel}")
            md_rel = _md_target(rel, used)
            try:
                images = load_file(rel.name, zf.read(info), dpi=dpi, pages=pages)
                with tempfile.TemporaryDirectory(prefix="uocr_") as tmp:
                    _place(run_ocr(images, tmp, model, tokenizer, **ocr_kw), out_dir, md_rel)
                res.done.append(str(md_rel))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                res.errors.append((str(rel), "sem memória na GPU (tente --quant 4bit ou menor --dpi)"))
            except Exception as e:  # um arquivo ruim não derruba o lote
                res.errors.append((str(rel), f"{type(e).__name__}: {e}"))
    res.seconds = time.time() - t0
    _write_report(res)
    if progress:
        progress(len(entries), len(entries), "concluído")
    return res


def make_zip(folder: str | Path) -> Path:
    """Compacta `folder` (conteúdo na raiz do zip) em `<folder>.zip`."""
    folder = Path(folder)
    return Path(shutil.make_archive(str(folder), "zip", root_dir=folder))
