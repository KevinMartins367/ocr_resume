"""uocr.services.resumo e services.base: validação antes do modelo, opções e lock da GPU."""
import threading
from unittest.mock import MagicMock

import pytest

from uocr.services import base
from uocr.services.resumo import SummaryOptions, SummaryService


def test_validates_before_loading_model(tmp_path):
    provider = MagicMock()
    service = SummaryService(provider, "q/7b")
    with pytest.raises(ValueError, match="não encontrada"):
        service.summarize(tmp_path / "nao-existe")
    (tmp_path / "vazia").mkdir()
    with pytest.raises(ValueError, match="nenhum .md"):
        service.summarize(tmp_path / "vazia")
    provider.assert_not_called()


def test_runs_with_options(tmp_path, fake_llm):
    (tmp_path / "cap1").mkdir()
    (tmp_path / "cap1" / "a.md").write_text("texto", encoding="utf-8")
    opts = SummaryOptions(max_input_tokens=900, max_new_tokens=77, instruction="foque em datas")
    res = SummaryService(lambda: fake_llm, "q/7b").summarize(tmp_path, opts)
    assert res.done == ["cap1"]
    call = fake_llm.calls[0]
    assert call["max_new_tokens"] == 77 and "foque em datas" in call["user"]
    assert "q/7b" in (tmp_path / "cap1" / "_resumo.md").read_text(encoding="utf-8")


def test_using_model_holds_gpu_lock():
    """Enquanto um serviço usa o modelo, outro (em outra thread) espera."""
    inside = []
    service = base.ModelService(lambda: "m")

    def other():
        with base.GPU_LOCK:
            inside.append("outro")

    with service.using_model() as m:
        assert m == "m"
        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=0.2)
        assert inside == []                    # bloqueado
    t.join(timeout=2)
    assert inside == ["outro"]
