"""uocr.web: handlers das abas com os serviços mockados e smoke test da interface."""
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import gradio as gr
import pytest

from uocr import web
from uocr.ocr.batch import BatchResult
from uocr.resumo.folders import SUMMARY_NAME, SummaryResult
from uocr.services.ocr import OcrOptions
from uocr.services.resumo import SummaryOptions


@pytest.fixture(autouse=True)
def no_vram(monkeypatch):
    monkeypatch.setattr(web, "vram_report", lambda: "vram")


@pytest.fixture
def ocr():
    def document(paths, out_dir, opts, progress=None):
        progress(0, 1, "janela 1/1")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "result.md").write_text("# ok")
        return SimpleNamespace(markdown="# ok", pages=2, tokens=7, seconds=1,
                               out_dir=Path(out_dir), box_images=[Path(out_dir) / "b.jpg"])

    def batch(path, out_dir, opts, progress=None):
        Path(out_dir, "pasta").mkdir(parents=True, exist_ok=True)
        Path(out_dir, "pasta", "a.md").write_text("x")
        return BatchResult(out_dir=Path(out_dir), done=["pasta/a.md", "b.md"],
                           errors=[("c.png", "erro")], skipped=["d.txt (formato não suportado)"])

    return MagicMock(**{"document.side_effect": document, "batch.side_effect": batch})


OPTS = OcrOptions()


def test_ocr_options_from_ui():
    opts = web.ocr_options(150.0, "", 3.0, "base", 500.0)
    assert opts == OcrOptions(dpi=150, pages=None, batch_pages=3, mode="base", tokens_per_page=500)


def test_document_requires_files(ocr, tmp_path):
    with pytest.raises(gr.Error):
        web.handle_document(ocr, tmp_path, [], OPTS)
    ocr.document.assert_not_called()


def test_document_flow(ocr, tmp_path):
    progress = MagicMock()
    md, raw, gallery, zip_path, status = web.handle_document(ocr, tmp_path, ["/x/doc.pdf"], OPTS, progress)
    assert (md, raw) == ("# ok", "# ok")
    assert gallery[0].endswith("b.jpg") and "2 página(s)" in status
    assert zipfile.is_zipfile(zip_path) and Path(zip_path).stem.endswith("_doc")
    assert ocr.document.call_args.args[0] == ["/x/doc.pdf"]
    progress.assert_called_once_with((0, 1), desc="janela 1/1")   # adaptador do gr.Progress


def test_document_invalid_input_becomes_gr_error(ocr, tmp_path):
    ocr.document.side_effect = ValueError("formato não suportado: x.txt")
    with pytest.raises(gr.Error):
        web.handle_document(ocr, tmp_path, ["x.txt"], OPTS)


def test_zip_rejects_non_zip(ocr, tmp_path):
    bad = tmp_path / "x.zip"
    bad.write_bytes(b"nao e zip")
    with pytest.raises(gr.Error):
        web.handle_zip(ocr, tmp_path, str(bad), OPTS)
    with pytest.raises(gr.Error):
        web.handle_zip(ocr, tmp_path, None, OPTS)
    ocr.batch.assert_not_called()


def test_zip_flow(ocr, tmp_path):
    src = tmp_path / "lote.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("pasta/a.png", b"x")
    summary, zip_path, status = web.handle_zip(ocr, tmp_path / "out", str(src), OPTS)
    assert "### Gerados (2)" in summary and "### Erros (1)" in summary and "### Ignorados (1)" in summary
    assert "2 .md gerado(s) · 1 erro(s) · 1 ignorado(s)" in status
    with zipfile.ZipFile(zip_path) as z:
        assert "pasta/a.md" in z.namelist()


def test_zip_value_error_becomes_gr_error(ocr, tmp_path):
    ocr.batch.side_effect = ValueError("o .zip não tem imagens nem PDFs")
    src = tmp_path / "vazio.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("a.txt", b"x")
    with pytest.raises(gr.Error):
        web.handle_zip(ocr, tmp_path / "out", str(src), OPTS)


def test_batch_summary_truncates():
    res = BatchResult(out_dir=Path("."), done=[f"{i}.md" for i in range(5)])
    text = web._batch_summary(res, limit=2)
    assert "### Gerados (5)" in text and "e mais 3" in text and "### Erros" not in text


# ---------------------------------------------------------------- Resumo

@pytest.fixture
def output(tmp_path):
    """output/ com um lote da CLI, um da web, um documento e uma pasta sem .md."""
    base = tmp_path / "output"
    for rel in ("lote/cap1/a.md", "web/20260101-000000_zip/p/b.md", "vazia/x.txt"):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return base


def test_summary_base():
    assert web.summary_base(Path("/a/output/web")) == Path("/a/output")
    assert web.summary_base(Path("/a/saida")) == Path("/a/saida")


def test_list_batches(output):
    assert sorted(web.list_batches(output)) == ["lote", "web/20260101-000000_zip"]


@pytest.fixture
def summary(output):
    def summarize(folder, opts, progress=None):
        (Path(folder) / "cap1" / SUMMARY_NAME).write_text("# Resumo: cap1\n\nok", encoding="utf-8")
        return SummaryResult(root=Path(folder), done=["cap1"], errors=[("cap2", "ruim")])
    return MagicMock(**{"summarize.side_effect": summarize})


def test_summary_flow(summary, output):
    text, zip_path, status = web.handle_summary(summary, output, "lote", SummaryOptions(force=True))
    assert "# Resumo: cap1" in text and "### Erros" in text and "`cap2`: ruim" in text
    assert "1 pasta(s) resumida(s)" in status and "1 erro(s)" in status
    assert summary.summarize.call_args.args[1].force is True
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["cap1/_resumo.md"]              # só os resumos


@pytest.mark.parametrize("choice", ["", None, "../fora", "nao-existe"])
def test_summary_rejects_bad_choice(summary, output, choice):
    with pytest.raises(gr.Error):
        web.handle_summary(summary, output, choice, SummaryOptions())
    summary.summarize.assert_not_called()


def test_summary_value_error_becomes_gr_error(summary, output):
    summary.summarize.side_effect = ValueError("nenhum .md")
    with pytest.raises(gr.Error):
        web.handle_summary(summary, output, "lote", SummaryOptions())


def test_build_app_smoke(output):
    app = web.build_app(MagicMock(), MagicMock(), output / "web", "info")
    assert isinstance(app, gr.Blocks)


def test_list_batches_without_web_folder(tmp_path):
    (tmp_path / "lote").mkdir()
    (tmp_path / "lote" / "a.md").write_text("x")
    assert web.list_batches(tmp_path) == ["lote"]


def test_wrappers_build_options(output):
    """Os wrappers do Gradio convertem os valores da interface e chamam os handlers."""
    ocr, summary = MagicMock(), MagicMock()
    app = web.build_app(ocr, summary, output / "web", "info")
    fns = {f.name: f.fn for f in app.fns.values()}
    calls = {}
    for name in ("handle_document", "handle_zip", "handle_summary"):
        calls[name] = MagicMock(return_value=name)
    with pytest.MonkeyPatch.context() as mp:
        for name, mock in calls.items():
            mp.setattr(web, name, mock)
        assert fns["run_document"](["a.pdf"], 150.0, "", 2.0, "base", 900.0, None) == "handle_document"
        assert fns["run_batch"]("l.zip", 200, "1", 4, "auto", 8000, None) == "handle_zip"
        assert fns["run_summary"]("lote", None, 1, 3000.0, 512.0, None) == "handle_summary"
        assert fns["refresh"]().choices                          # lista os lotes de output/
    assert calls["handle_document"].call_args.args[3] == OcrOptions(dpi=150, batch_pages=2, mode="base",
                                                                    tokens_per_page=900)
    assert calls["handle_zip"].call_args.args[3].pages == "1"
    assert calls["handle_summary"].call_args.args[3] == SummaryOptions(
        max_input_tokens=3000, max_new_tokens=512, instruction="", force=True)


def test_main_wires_services(monkeypatch, tmp_path):
    slot = MagicMock()
    app = MagicMock()
    monkeypatch.setattr(web, "OcrAdapter", MagicMock(return_value="ocr-adapter"))
    monkeypatch.setattr(web, "QwenAdapter", MagicMock(return_value="qwen-adapter"))
    monkeypatch.setattr(web, "GpuSlot", MagicMock(return_value=slot))
    build = MagicMock(return_value=app)
    monkeypatch.setattr(web, "build_app", build)

    web.main(["--output", str(tmp_path / "output" / "web"), "--port", "7999", "--summary-quant", "8bit"])

    web.QwenAdapter.assert_called_once_with("Qwen/Qwen2.5-7B-Instruct", "8bit")
    slot.get.assert_called_once_with("ocr-adapter")              # OCR carregado na partida
    ocr_svc, sum_svc, out_root, info = build.call_args.args
    assert sum_svc.model_id == "Qwen/Qwen2.5-7B-Instruct" and out_root.name == "web"
    kw = app.queue.return_value.launch.call_args.kwargs
    assert kw["server_port"] == 7999 and kw["allowed_paths"] == [str((tmp_path / "output").resolve())]
