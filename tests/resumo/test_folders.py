"""uocr.resumo.folders: um _resumo.md por pasta, sem resumo geral."""
import os

import pytest
import torch

from conftest import FakeLLM
from uocr.resumo import folders as f


@pytest.fixture
def lote(tmp_path):
    """lote/
         solto.md               (a raiz tem .md: ela também é uma pasta resumida)
         cap10/a.md
         cap2/pag2.md, pag10.md, pag2_images/x.md (figuras: ignorado)
         vazio/sub/b.md         (vazio/ não tem .md: não ganha resumo)
         .oculta/c.md           (ignorado)
    """
    root = tmp_path / "lote"
    for rel in ("solto.md", "cap10/a.md", "cap2/pag2.md", "cap2/pag10.md", "cap2/pag2_images/x.md",
                "vazio/sub/b.md", ".oculta/c.md"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"conteúdo de {rel}", encoding="utf-8")
    return root


def labels(found, root):
    return [d.relative_to(root).as_posix() for d, _ in found]


def test_find_folders(lote):
    found = f.find_folders(lote)
    assert labels(found, lote) == [".", "cap2", "cap10", "vazio/sub"]          # ordem natural
    assert [p.name for p in dict(found)[lote / "cap2"]] == ["pag2.md", "pag10.md"]


def test_one_summary_per_folder_no_general(lote, fake_llm):
    progress = []
    res = f.summarize_tree(lote, fake_llm, model_id="q/7b", progress=lambda *a: progress.append(a))
    assert res.done == ["lote", "cap2", "cap10", "vazio/sub"] and not res.errors
    assert not (lote / "vazio" / f.SUMMARY_NAME).exists()                       # sem resumo da mãe
    text = (lote / "cap2" / f.SUMMARY_NAME).read_text(encoding="utf-8")
    assert text.startswith("# Resumo: cap2") and "q/7b" in text
    assert "Arquivos (2): `pag2.md`, `pag10.md`" in text
    root_call = fake_llm.calls[0]["user"]
    assert "## solto.md" in root_call and "cap2" not in root_call               # só os .md da raiz
    assert progress[-1][2] == "concluído"


def test_up_to_date_is_skipped_unless_forced(lote, fake_llm):
    f.summarize_tree(lote, fake_llm)
    n = len(fake_llm.calls)
    res = f.summarize_tree(lote, fake_llm)
    assert res.done == [] and len(res.up_to_date) == 4 and len(fake_llm.calls) == n

    md = lote / "cap2" / "pag2.md"                                              # um .md mudou
    future = (lote / "cap2" / f.SUMMARY_NAME).stat().st_mtime + 10
    os.utime(md, (future, future))
    assert f.summarize_tree(lote, fake_llm).done == ["cap2"]
    assert len(f.summarize_tree(lote, fake_llm, force=True).done) == 4


def test_errors_dont_stop_other_folders(lote):
    llm = FakeLLM(fail_on="cap10/a.md", fail_with=ValueError("ruim"))
    res = f.summarize_tree(lote, llm)
    assert res.errors == [("cap10", "ValueError: ruim")]
    assert "cap2" in res.done and not (lote / "cap10" / f.SUMMARY_NAME).exists()


def test_out_of_memory(lote, cuda_empty_cache):
    llm = FakeLLM(fail_on="cap2/pag2.md", fail_with=torch.cuda.OutOfMemoryError("oom"))
    res = f.summarize_tree(lote, llm)
    assert ("cap2", "sem memória na GPU (tente --max-input-tokens menor)") in res.errors
    cuda_empty_cache.assert_called()


def test_invalid_roots(tmp_path, fake_llm):
    with pytest.raises(ValueError, match="não encontrada"):
        f.summarize_tree(tmp_path / "x", fake_llm)
    with pytest.raises(ValueError, match="nenhum .md"):
        f.summarize_tree(tmp_path, fake_llm)


def test_collect_summaries(lote, fake_llm):
    f.summarize_tree(lote, fake_llm)
    text = f.collect_summaries(lote)
    order = [line for line in text.splitlines() if line.startswith("# Resumo:")]
    assert order == ["# Resumo: lote", "# Resumo: cap2", "# Resumo: cap10", "# Resumo: vazio/sub"]
