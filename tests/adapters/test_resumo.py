"""uocr.adapters.resumo: carga do Qwen (mockada), embeddings na CPU e o Summarizer."""
from unittest.mock import MagicMock

import pytest
import torch

from uocr.adapters import resumo
from uocr.adapters.resumo import QwenAdapter, Summarizer


@pytest.fixture
def emb_cpu(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(resumo, "embeddings_on_cpu", mock)
    return mock


def _kw(hf):
    return hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs


def test_4bit_quantizes_lm_head_and_moves_embeddings(hf, emb_cpu):
    llm = QwenAdapter("q/7b", "4bit").load()
    assert isinstance(llm, Summarizer) and llm.model is hf.model
    assert hf.AutoModelForCausalLM.from_pretrained.call_args.args[0] == "/cache/q/7b"
    _, bnb = _kw(hf)["quantization_config"]
    assert bnb["load_in_4bit"] and bnb["llm_int8_skip_modules"] == []   # lm_head também em 4 bits
    emb_cpu.assert_called_once_with(hf.model)


def test_tied_embeddings_keep_default(hf, emb_cpu):
    hf.config.tie_word_embeddings = True               # Qwen2.5 ≤ 3B
    QwenAdapter("q/small", "4bit").load()
    _, bnb = _kw(hf)["quantization_config"]
    assert bnb["llm_int8_skip_modules"] is None
    emb_cpu.assert_not_called()


def test_8bit_and_bf16(hf, emb_cpu):
    QwenAdapter("q/7b", "8bit").load()
    _, bnb = _kw(hf)["quantization_config"]
    assert bnb["load_in_8bit"] and bnb["llm_int8_skip_modules"] is None
    QwenAdapter("q/7b", "bf16").load()
    assert "quantization_config" not in _kw(hf)
    emb_cpu.assert_not_called()


def test_invalid_quant():
    with pytest.raises(ValueError, match="quant inválido"):
        QwenAdapter("q/7b", "offload")


def test_embeddings_on_cpu_routes_through_cpu():
    """Ids entram na CPU e os vetores voltam para o dispositivo das outras camadas."""
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 4)
            self.proj = torch.nn.Linear(4, 2)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, ids):
            return self.proj(self.embed_tokens(ids))

    m = Tiny()
    seen = []
    m.embed_tokens.register_forward_pre_hook(lambda _m, args: seen.append(args[0].device))
    resumo.embeddings_on_cpu(m)
    out = m(torch.tensor([[1, 2, 3]]))
    assert m.embed_tokens.weight.device.type == "cpu" and seen == [torch.device("cpu")]
    assert out.shape == (1, 3, 2) and out.device == m.proj.weight.device


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, msgs, add_generation_prompt, return_tensors):
        self.msgs = msgs
        return torch.tensor([[5, 6, 7]])

    def decode(self, ids, skip_special_tokens):
        self.decoded = ids.tolist()
        return "```markdown\n# Resumo\n- ponto\n```"


def test_summarizer_count_and_generate():
    tok = FakeTokenizer()
    model = MagicMock()
    model.generate.return_value = torch.tensor([[5, 6, 7, 8, 9]])
    llm = Summarizer(model, tok, device="cpu")

    assert llm.count("três palavras aqui") == 3
    out = llm.generate("sistema", "usuário", 50)
    assert out == "# Resumo\n- ponto"                                   # cerca ``` removida
    assert tok.msgs == [{"role": "system", "content": "sistema"}, {"role": "user", "content": "usuário"}]
    assert tok.decoded == [8, 9]                                         # só os tokens novos
    kw = model.generate.call_args.kwargs
    assert kw["max_new_tokens"] == 50 and kw["do_sample"] is False and kw["streamer"] is None
