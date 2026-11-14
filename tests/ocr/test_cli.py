"""uocr.ocr.cli: roteamento dos argumentos com adapter e serviço mockados."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from uocr.ocr import cli


@pytest.fixture
def mocks(monkeypatch):
    service = MagicMock()
    service.document.side_effect = lambda files, out_dir, *a: SimpleNamespace(
        out_dir=Path(out_dir), pages=len(files), tokens=1, seconds=0)
    service.batch.side_effect = lambda z, out_dir, *a: SimpleNamespace(
        out_dir=Path(out_dir), done=["a.md"], errors=[], skipped=[], seconds=0)
    ns = SimpleNamespace(service=service, Service=MagicMock(return_value=service),
                         Adapter=MagicMock(name="OcrAdapter"))
    monkeypatch.setattr("uocr.services.ocr.OcrService", ns.Service)
    monkeypatch.setattr("uocr.adapters.ocr.OcrAdapter", ns.Adapter)
    monkeypatch.setattr("uocr.core.gpu.vram_report", lambda: "vram")
    return ns


@pytest.fixture
def files(tmp_path):
    for name in ("a.png", "a.pdf", "b.jpg", "lote.zip"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def out_dirs(mock):
    return [Path(c.args[1]).name for c in mock.call_args_list]


def test_parser_defaults():
    a = cli.build_parser().parse_args(["x.pdf"])
    assert (a.quant, a.dpi, a.batch_pages, a.mode, a.output) == ("4bit", 200, 4, "auto", "output")


def test_missing_file(mocks, tmp_path):
    with pytest.raises(SystemExit, match="não encontrado"):
        cli.main([str(tmp_path / "nao-existe.pdf")])
    mocks.Adapter.assert_not_called()


def test_each_file_is_a_document_and_names_dont_collide(mocks, files, tmp_path):
    out = tmp_path / "out"
    assert cli.main([str(files / "a.png"), str(files / "a.pdf"), "-o", str(out), "--quant", "8bit"]) == 0
    mocks.Adapter.assert_called_once_with("baidu/Unlimited-OCR-2", "8bit", 6)
    assert out_dirs(mocks.service.document) == ["a", "a_pdf"]


def test_combine_and_options(mocks, files):
    cli.main([str(files / "a.png"), str(files / "b.jpg"), "--combine", "--batch-pages", "2"])
    assert mocks.service.document.call_count == 1
    files_arg, _, opts, _ = mocks.service.document.call_args.args
    assert len(files_arg) == 2 and opts.batch_pages == 2


def test_zip_goes_to_batch(mocks, files, tmp_path):
    out = tmp_path / "out"
    cli.main([str(files / "lote.zip"), str(files / "b.jpg"), "-o", str(out), "--dpi", "150",
              "--mode", "base"])
    assert out_dirs(mocks.service.batch) == ["lote"]
    opts = mocks.service.batch.call_args.args[2]
    assert (opts.dpi, opts.mode) == (150, "base")
    assert out_dirs(mocks.service.document) == ["b"]          # o resto segue o fluxo normal
