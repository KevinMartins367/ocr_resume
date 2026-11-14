"""uocr.adapters.ocr: carga do Unlimited-OCR-2 com transformers/bitsandbytes mockados."""
import pytest

from uocr.adapters.ocr import SKIP_MODULES, OcrAdapter


def _call(hf):
    return hf.AutoModel.from_pretrained.call_args


@pytest.mark.parametrize("quant, flag", [("4bit", "load_in_4bit"), ("8bit", "load_in_8bit")])
def test_quantized_load(hf, quant, flag):
    model, tok = OcrAdapter("x/y", quant).load()
    call = _call(hf)
    kind, bnb_kw = call.kwargs["quantization_config"]
    assert (model, tok, kind) == (hf.model, "tok", "bnb")
    assert call.args[0] == "/cache/x/y"                         # carrega da pasta local (offline)
    assert bnb_kw[flag] is True
    assert bnb_kw["llm_int8_skip_modules"] == SKIP_MODULES     # visão/atenção ficam em bf16
    assert call.kwargs["device_map"] == {"": 0} and call.kwargs["trust_remote_code"] is True
    hf.model.cuda.assert_not_called()


def test_bf16_moves_to_gpu(hf):
    OcrAdapter("x/y", "bf16").load()
    assert "quantization_config" not in _call(hf).kwargs
    hf.model.cuda.assert_called_once()


def test_offload_device_map(hf):
    OcrAdapter("x/y", "offload", offload_layers=2).load()
    dmap = _call(hf).kwargs["device_map"]
    assert dmap.pop("") == 0
    expected = {f"model.layers.{li}.mlp.experts.{e}" for li in (2, 3) for e in range(3)}
    assert set(dmap) == expected and set(dmap.values()) == {"cpu"}


def test_load_is_cached(hf):
    a = OcrAdapter("x/y", "4bit")
    assert a.load()[0] is a.load()[0]
    assert hf.AutoModel.from_pretrained.call_count == 1


def test_invalid_quant():
    with pytest.raises(ValueError, match="quant inválido"):
        OcrAdapter("x/y", "2bit")
