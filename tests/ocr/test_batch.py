"""uocr.ocr.batch: leitura segura do zip e espelhamento da estrutura (run_ocr mockado)."""
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import torch

from uocr.ocr import batch
from uocr.ocr.batch import _decode_name, _md_target, _safe_rel, list_entries, make_zip, run_zip


def _zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


@pytest.fixture
def fake_run_ocr(monkeypatch):
    """Substitui o OCR: grava uma figura em images/ e devolve Markdown que a referencia."""
    calls = []

    def _fake(images, out_dir, model, tokenizer, **kw):
        calls.append({"pages": len(images), **kw})
        (Path(out_dir) / "images").mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "images" / "page1_0_0_10_10.jpg").write_bytes(b"jpg")
        md = f"# Documento com {len(images)} página(s)\n\n![](images/page1_0_0_10_10.jpg)\n"
        return SimpleNamespace(markdown=md, out_dir=Path(out_dir))

    monkeypatch.setattr(batch, "run_ocr", _fake)
    return calls


# ---------- nomes e caminhos ----------

def test_decode_name_utf8_flag():
    info = zipfile.ZipInfo("Relatório/página.png")
    info.flag_bits |= 0x800
    assert _decode_name(info) == "Relatório/página.png"


@pytest.mark.parametrize("encoding", ["cp850", "utf-8"])
def test_decode_name_legacy_windows(encoding):
    """Sem a flag UTF-8, o zipfile decodifica como cp437; recuperamos o nome original."""
    raw = "Relatório/página.png".encode(encoding)
    info = zipfile.ZipInfo(raw.decode("cp437"))
    info.flag_bits = 0
    assert _decode_name(info) == "Relatório/página.png"


@pytest.mark.parametrize("name, expected", [
    ("a/b.png", "a/b.png"),
    ("a\\b\\c.png", "a/b/c.png"),
    ("./a/./b.png", "a/b.png"),
    ("/abs/x.png", "abs/x.png"),
    ("../x.png", None),
    ("a/../../x.png", None),
    ("C:/x.png", None),
    ("", None),
])
def test_safe_rel(name, expected):
    rel = _safe_rel(name)
    assert (str(rel) if rel else None) == expected


def test_md_target_collision_is_case_insensitive():
    used: set[str] = set()
    assert str(_md_target(PurePosixPath("a/foto.jpg"), used)) == "a/foto.md"
    assert str(_md_target(PurePosixPath("a/FOTO.png"), used)) == "a/FOTO_png.md"


# ---------- list_entries ----------

def test_list_entries_filters_and_sorts(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "MAX_MEMBER_BYTES", 5)
    z = _zip(tmp_path / "in.zip", {
        "p/b10.png": b"x", "p/b2.png": b"x", "p/a.pdf": b"x",
        "p/notas.txt": b"x", "p/grande.png": b"x" * 10,
        "../evil.png": b"x", "__MACOSX/p/._b2.png": b"x", "p/.DS_Store": b"x", "p/Thumbs.db": b"x",
    })
    with zipfile.ZipFile(z) as zf:
        entries, skipped = list_entries(zf)
    assert [str(rel) for rel, _ in entries] == ["p/a.pdf", "p/b2.png", "p/b10.png"]
    assert sorted(skipped) == ["../evil.png (caminho inválido)",
                               "p/grande.png (maior que 0 MB)",
                               "p/notas.txt (formato não suportado)"]


def test_list_entries_max_files(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "MAX_FILES", 1)
    z = _zip(tmp_path / "in.zip", {"a.png": b"x", "b.png": b"x"})
    with zipfile.ZipFile(z) as zf, pytest.raises(ValueError, match="máximo 1"):
        list_entries(zf)


# ---------- run_zip ----------

def test_run_zip_mirrors_structure(tmp_path, png_bytes, pdf_bytes, fake_run_ocr):
    z = _zip(tmp_path / "in.zip", {
        "Capítulo 1/pag 1.png": png_bytes,
        "Capítulo 1/anexos/contrato.pdf": pdf_bytes,
        "raiz.png": png_bytes,
        "leia-me.txt": b"x",
    })
    events = []
    res = run_zip(z, tmp_path / "out", None, None, dpi=72, mode="base",
                  progress=lambda i, n, m: events.append((i, n)))

    out = tmp_path / "out"
    assert sorted(res.done) == ["Capítulo 1/anexos/contrato.md", "Capítulo 1/pag 1.md", "raiz.md"]
    md = (out / "Capítulo 1" / "pag 1.md").read_text(encoding="utf-8")
    assert "](pag%201_images/page1_0_0_10_10.jpg)" in md      # link reescrito e com URL-encode
    assert (out / "Capítulo 1" / "pag 1_images" / "page1_0_0_10_10.jpg").exists()
    assert "3 página(s)" in (out / "Capítulo 1" / "anexos" / "contrato.md").read_text(encoding="utf-8")
    assert fake_run_ocr[0]["mode"] == "base"                    # kwargs repassados ao OCR
    assert res.skipped == ["leia-me.txt (formato não suportado)"]
    assert "leia-me.txt" in (out / batch.REPORT_NAME).read_text(encoding="utf-8")
    assert events[-1] == (3, 3)


def test_run_zip_isolates_errors(tmp_path, png_bytes, fake_run_ocr, monkeypatch, cuda_empty_cache):
    real = batch.run_ocr

    def oom_on_second(images, out_dir, *a, **kw):
        if len(fake_run_ocr) == 1:
            fake_run_ocr.append({})
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(images, out_dir, *a, **kw)

    monkeypatch.setattr(batch, "run_ocr", oom_on_second)
    z = _zip(tmp_path / "in.zip", {"1.png": png_bytes, "2.png": png_bytes,
                                   "3-corrompida.png": b"nao e imagem"})
    res = run_zip(z, tmp_path / "out", None, None)

    assert res.done == ["1.md"]
    errors = dict(res.errors)
    assert "sem memória" in errors["2.png"]
    assert "UnidentifiedImageError" in errors["3-corrompida.png"]
    cuda_empty_cache.assert_called()


def test_run_zip_no_report_when_clean(tmp_path, png_bytes, fake_run_ocr):
    z = _zip(tmp_path / "in.zip", {"a.png": png_bytes})
    run_zip(z, tmp_path / "out", None, None)
    assert not (tmp_path / "out" / batch.REPORT_NAME).exists()


def test_run_zip_without_supported_files(tmp_path, fake_run_ocr):
    z = _zip(tmp_path / "in.zip", {"a.txt": b"x"})
    with pytest.raises(ValueError, match="não tem imagens"):
        run_zip(z, tmp_path / "out", None, None)


def test_make_zip_has_folder_content_at_root(tmp_path):
    folder = tmp_path / "job"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.md").write_text("x")
    with zipfile.ZipFile(make_zip(folder)) as z:
        assert "sub/a.md" in z.namelist()


def test_list_entries_skips_directory_entries(tmp_path, png_bytes):
    z = _zip(tmp_path / "d.zip", {"pasta/": b"", "pasta\\": b"", "pasta/a.png": png_bytes})
    with zipfile.ZipFile(z) as zf:
        entries, skipped = list_entries(zf)
    assert [str(rel) for rel, _ in entries] == ["pasta/a.png"] and skipped == []
