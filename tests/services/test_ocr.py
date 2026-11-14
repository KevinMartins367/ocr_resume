"""uocr.services.ocr: valida antes de carregar o modelo e repassa as opções ao pipeline."""
from unittest.mock import MagicMock

import pytest

from uocr.services import ocr as svc
from uocr.services.ocr import OcrOptions, OcrService


@pytest.fixture
def mocks(monkeypatch):
    m = MagicMock()
    m.load_inputs.return_value = ["img1", "img2"]
    m.provider.return_value = ("model", "tok")
    monkeypatch.setattr(svc, "load_inputs", m.load_inputs)
    monkeypatch.setattr(svc, "run_ocr", m.run_ocr)
    monkeypatch.setattr(svc, "run_zip", m.run_zip)
    return m


OPTS = OcrOptions(dpi=150, pages="1-2", batch_pages=3, mode="base", tokens_per_page=500, stream=True)


def test_document(mocks, tmp_path):
    progress = MagicMock()
    OcrService(mocks.provider).document(["a.pdf"], tmp_path, OPTS, progress)
    mocks.load_inputs.assert_called_once_with(["a.pdf"], dpi=150, pages="1-2")
    args, kw = mocks.run_ocr.call_args
    assert args == (["img1", "img2"], tmp_path, "model", "tok")
    assert kw == dict(progress=progress, batch_pages=3, mode="base", tokens_per_page=500, stream=True)


def test_document_validates_before_loading_model(mocks, tmp_path):
    mocks.load_inputs.side_effect = ValueError("formato não suportado: x.txt")
    with pytest.raises(ValueError):
        OcrService(mocks.provider).document(["x.txt"], tmp_path)
    mocks.provider.assert_not_called()


def test_batch(mocks, tmp_path):
    OcrService(mocks.provider).batch("l.zip", tmp_path, OPTS)
    args, kw = mocks.run_zip.call_args
    assert args == ("l.zip", tmp_path, "model", "tok")
    assert (kw["dpi"], kw["pages"], kw["batch_pages"], kw["mode"]) == (150, "1-2", 3, "base")


def test_defaults():
    assert OcrOptions().pipeline_kw() == dict(batch_pages=4, mode="auto", tokens_per_page=8000, stream=False)
