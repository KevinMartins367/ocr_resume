"""Fixtures compartilhadas. Nenhum teste carrega o modelo real nem usa a GPU."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from PIL import Image


class FakeModel:
    """Imita `UnlimitedOCRForCausalLM.infer(..., return_dict=True)`.

    Devolve, por página, blocos no formato real do Unlimited-OCR-2:
    `<PAGE><BLOCK>title<LEVEL1>[..]<FORK>... <BLOCK>text[..]<FORK>... <BLOCK>image[..]<FORK>`
    """

    def __init__(self, pages_out: int | None = None, prefix_page: bool = True):
        self.calls: list[dict] = []
        self.pages_out = pages_out      # força o nº de páginas devolvidas (testa o aviso)
        self.prefix_page = prefix_page  # False: resposta sem o <PAGE> inicial

    def infer(self, tokenizer, image_file, **kw):
        self.calls.append({"n": len(image_file), "images": image_file, **kw})
        print("[DASH] debug do modelo")  # o pipeline deve esconder isso fora do --stream
        start = sum(c["n"] for c in self.calls[:-1])
        n = self.pages_out if self.pages_out is not None else len(image_file)
        text = "".join(
            f"<PAGE><BLOCK>title<LEVEL1>[100,50,900,100]<FORK>Título {start + k + 1}"
            f"<BLOCK>text[100,150,900,300]<FORK>Texto da página {start + k + 1} com acentuação."
            f"<BLOCK>image[200,400,600,800]<FORK>"
            for k in range(n))
        if not self.prefix_page:
            text = text[len("<PAGE>"):]
        return {"text": text, "token_ids": list(range(10 * n)), "num_pages_out": n,
                "finish_reason": "stop"}


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def make_image():
    def _make(w: int = 400, h: int = 600, color=(255, 255, 255)) -> Image.Image:
        return Image.new("RGB", (w, h), color)
    return _make


@pytest.fixture
def png_bytes(make_image) -> bytes:
    buf = io.BytesIO()
    make_image(50, 80).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    """PDF de 3 páginas de 200x300 pt."""
    import fitz

    doc = fitz.open()
    for i in range(3):
        doc.new_page(width=200, height=300).insert_text((20, 40), f"Página {i + 1}", fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def cuda_empty_cache(monkeypatch):
    """Mock de torch.cuda.empty_cache: não inicializa CUDA e permite contar as chamadas."""
    mock = MagicMock()
    monkeypatch.setattr("torch.cuda.empty_cache", mock)
    return mock


@pytest.fixture
def hf(monkeypatch):
    """transformers mockado, GPU "disponível" e Hub offline (o id vira /cache/<id>).

    `hf.config` é o que o AutoConfig devolve: ajuste `tie_word_embeddings` ou
    `language_config` no teste.
    """
    from types import SimpleNamespace

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("uocr.core.hub.local_model_path", lambda mid: f"/cache/{mid}")
    model = MagicMock(name="model")
    model.eval.return_value = model
    model.cuda.return_value = model
    config = SimpleNamespace(tie_word_embeddings=False, language_config={
        "num_hidden_layers": 4, "n_routed_experts": 3, "first_k_dense_replace": 1})
    ns = SimpleNamespace(
        model=model, config=config,
        AutoModel=MagicMock(**{"from_pretrained.return_value": model}),
        AutoModelForCausalLM=MagicMock(**{"from_pretrained.return_value": model}),
        AutoTokenizer=MagicMock(**{"from_pretrained.return_value": "tok"}),
        BitsAndBytesConfig=MagicMock(side_effect=lambda **kw: ("bnb", kw)),
        AutoConfig=MagicMock(**{"from_pretrained.return_value": config}),
    )
    for name in ("AutoModel", "AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig", "AutoConfig"):
        monkeypatch.setattr(f"transformers.{name}", getattr(ns, name))
    return ns


class FakeLLM:
    """Imita o `Summarizer`: conta "tokens" como palavras e devolve um resumo curto numerado.

    `fail_on`: se o texto do prompt contiver essa string, `generate` levanta `fail_with`.
    """

    def __init__(self, fail_on: str | None = None, fail_with: Exception | None = None):
        self.calls: list[dict] = []
        self.fail_on, self.fail_with = fail_on, fail_with

    def count(self, text: str) -> int:
        return len(text.split())

    def generate(self, system: str, user: str, max_new_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "max_new_tokens": max_new_tokens})
        if self.fail_on and self.fail_on in user:
            raise self.fail_with or RuntimeError("falhou")
        return f"resumo {len(self.calls)}"


@pytest.fixture
def fake_llm():
    return FakeLLM()
