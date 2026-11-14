"""Percorre a saída do OCR e grava um `_resumo.md` em cada pasta que tenha .md."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import torch

from ..core.util import Progress, natural_key
from ..adapters.resumo import DEFAULT_SUMMARY_MODEL
from .text import DEFAULT_MAX_INPUT, DEFAULT_MAX_NEW, TextModel, summarize_texts

SUMMARY_NAME = "_resumo.md"


@dataclass
class SummaryResult:
    root: Path
    done: list[str] = field(default_factory=list)                    # pastas resumidas (relativas)
    up_to_date: list[str] = field(default_factory=list)              # já tinham _resumo.md atual
    errors: list[tuple[str, str]] = field(default_factory=list)      # (pasta, erro)
    seconds: float = 0.0


def find_folders(root: str | Path) -> list[tuple[Path, list[Path]]]:
    """[(pasta, .md diretos dela)] para cada pasta com ao menos um .md (fora o _resumo.md).
    Inclui a própria raiz se ela tiver .md soltos. Ignora pastas ocultas e de figuras."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith("_images")]
        mds = sorted((f for f in filenames if f.lower().endswith(".md") and f != SUMMARY_NAME),
                     key=natural_key)
        if mds:
            out.append((Path(dirpath), [Path(dirpath) / f for f in mds]))
    out.sort(key=lambda e: natural_key(e[0].relative_to(root).as_posix()))
    return out


def is_up_to_date(folder: Path, mds: list[Path]) -> bool:
    """O `_resumo.md` existe e é mais novo que todos os .md da pasta."""
    target = folder / SUMMARY_NAME
    return target.is_file() and target.stat().st_mtime >= max(p.stat().st_mtime for p in mds)


def write_summary(folder: Path, label: str, mds: list[Path], summary: str, model_id: str) -> Path:
    files = ", ".join(f"`{p.name}`" for p in mds)
    text = (f"# Resumo: {label}\n\n"
            f"<!-- gerado por uocr-resumo · {model_id} -->\n"
            f"Arquivos ({len(mds)}): {files}\n\n{summary.strip()}\n")
    target = folder / SUMMARY_NAME
    target.write_text(text, encoding="utf-8")
    return target


def summarize_tree(root: str | Path, llm: TextModel, *, model_id: str = DEFAULT_SUMMARY_MODEL,
                   max_input_tokens: int = DEFAULT_MAX_INPUT, max_new_tokens: int = DEFAULT_MAX_NEW,
                   instruction: str = "", force: bool = False,
                   progress: Progress | None = None) -> SummaryResult:
    """Grava um `_resumo.md` em cada pasta de `root` que tenha .md. Pastas cujo resumo é mais
    novo que todos os .md são puladas (a não ser com `force`). Um erro não interrompe as outras."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"pasta não encontrada: {root}")
    folders = find_folders(root)
    if not folders:
        raise ValueError(f"nenhum .md em {root}")
    res = SummaryResult(root=root)
    t0 = time.time()
    for i, (folder, mds) in enumerate(folders):
        rel = folder.relative_to(root).as_posix()
        label = root.name if rel == "." else rel
        if not force and is_up_to_date(folder, mds):
            res.up_to_date.append(label)
            continue
        if progress:
            progress(i, len(folders), f"{i + 1}/{len(folders)}: {label} ({len(mds)} arquivo(s))")
        try:
            docs = [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in mds]
            summary = summarize_texts(label, docs, llm, max_input_tokens=max_input_tokens,
                                      max_new_tokens=max_new_tokens, instruction=instruction)
            write_summary(folder, label, mds, summary, model_id)
            res.done.append(label)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res.errors.append((label, "sem memória na GPU (tente --max-input-tokens menor)"))
        except Exception as e:  # uma pasta ruim não derruba as outras
            res.errors.append((label, f"{type(e).__name__}: {e}"))
    res.seconds = time.time() - t0
    if progress:
        progress(len(folders), len(folders), "concluído")
    return res


def collect_summaries(root: str | Path) -> str:
    """Junta os `_resumo.md` de `root` (ordem natural das pastas), para exibir na web."""
    root = Path(root)
    found = sorted(root.rglob(SUMMARY_NAME), key=lambda p: natural_key(p.relative_to(root).as_posix()))
    return "\n\n---\n\n".join(p.read_text(encoding="utf-8").strip() for p in found)
