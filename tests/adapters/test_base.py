"""uocr.adapters.base: ciclo de vida do ModelAdapter, GpuSlot (um modelo por vez) e bnb_config."""
from unittest.mock import MagicMock

import pytest

from uocr.adapters import base


class FakeAdapter(base.ModelAdapter):
    kind = "fake"
    quant_choices = ("a", "b")

    def __init__(self, model_id="x/y", quant="a"):
        super().__init__(model_id, quant)
        self.builds: list[str] = []

    def _build(self, path):
        self.builds.append(path)
        return object()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("uocr.core.hub.local_model_path", lambda mid: f"/cache/{mid}")
    free = MagicMock()
    monkeypatch.setattr(base, "free", free)
    monkeypatch.setattr(base, "vram_report", lambda: "vram")
    return free


def test_load_resolves_local_path_and_caches(env):
    a = FakeAdapter()
    first, second = a.load(), a.load()
    assert first is second and a.builds == ["/cache/x/y"] and a.loaded


def test_unload_frees_only_when_loaded(env):
    a = FakeAdapter()
    a.unload()
    env.assert_not_called()
    a.load()
    a.unload()
    assert not a.loaded and env.call_count == 1
    a.load()
    assert len(a.builds) == 2                        # descarregado = carrega de novo


def test_invalid_quant():
    with pytest.raises(ValueError, match="quant inválido"):
        FakeAdapter(quant="z")


def test_requires_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        FakeAdapter().load()


def test_repr():
    assert repr(FakeAdapter("m/x", "b")) == "fake m/x (b)"


def test_gpu_slot_keeps_one_model(env):
    a, b = FakeAdapter("m/a"), FakeAdapter("m/b")
    log = MagicMock()
    slot = base.GpuSlot(a, b, log=log)

    slot.get(a)
    assert a.loaded and not b.loaded
    slot.get(b)                                      # troca: descarrega a, carrega b
    assert b.loaded and not a.loaded
    slot.get(b)                                      # já carregado: nada muda
    assert len(b.builds) == 1
    assert ">> Carregando fake m/b (a)..." in [c.args[0] for c in log.call_args_list]


def test_gpu_slot_provider(env):
    a = FakeAdapter()
    get = base.GpuSlot(a, log=lambda *_: None).provider(a)
    assert get() is a.load()


def test_bnb_config(hf):
    kind, kw = base.bnb_config("4bit", ["lm_head"])
    assert kind == "bnb" and kw["load_in_4bit"] and kw["bnb_4bit_quant_type"] == "nf4"
    assert kw["llm_int8_skip_modules"] == ["lm_head"]
    assert base.bnb_config("8bit")[1]["load_in_8bit"] is True
    assert base.bnb_config("bf16") is None and base.bnb_config("offload") is None
