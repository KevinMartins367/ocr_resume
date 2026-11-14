"""Adapter do Qwen2.5-7B-Instruct para uma GPU de 6 GB.

4 bits (NF4) em tudo, inclusive no lm_head, e a tabela de embeddings (~1 GB) na CPU:
sobram ~2 GB para o contexto. Não cabe junto com o modelo de OCR (ver `base.GpuSlot`).
"""
from __future__ import annotations

import os

import torch

from ..resumo.text import strip_fence
from .base import ModelAdapter, bnb_config

DEFAULT_SUMMARY_MODEL = os.environ.get("UOCR_SUMMARY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SUMMARY_QUANT_CHOICES = ("4bit", "8bit", "bf16")


def embeddings_on_cpu(model) -> None:
    """Move a tabela de embeddings para a CPU, onde a busca por id é barata. Os ids vão
    para a CPU antes da busca e os vetores voltam para a GPU depois."""
    emb = model.get_input_embeddings()
    gpu = next(p.device for n, p in model.named_parameters() if not n.endswith("embed_tokens.weight"))
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda _m, args: tuple(
        a.to("cpu") if torch.is_tensor(a) else a for a in args))
    emb.register_forward_hook(lambda _m, _args, out: out.to(gpu))
    torch.cuda.empty_cache()


class Summarizer:
    """Modelo de chat pronto para o resumo (interface `resumo.text.TextModel`): conta tokens e gera."""

    def __init__(self, model, tokenizer, stream: bool = False, device: str = "cuda"):
        # `device`: onde ficam os ids de entrada (as camadas estão na GPU; os embeddings, na CPU,
        # recebem os ids pelo hook de `embeddings_on_cpu`).
        self.model, self.tokenizer, self.stream, self.device = model, tokenizer, stream, device

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, system: str, user: str, max_new_tokens: int) -> str:
        from transformers import TextStreamer

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.stream else None
        with torch.inference_mode():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer)
        text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        del ids, out
        torch.cuda.empty_cache()
        return strip_fence(text)


class QwenAdapter(ModelAdapter):
    """`load()` devolve um `Summarizer`."""

    kind = "resumo"
    quant_choices = SUMMARY_QUANT_CHOICES

    def __init__(self, model_id: str = DEFAULT_SUMMARY_MODEL, quant: str = "4bit", stream: bool = False):
        super().__init__(model_id, quant)
        self.stream = stream

    def _build(self, path: str) -> Summarizer:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        # Modelos menores (Qwen2.5 ≤ 3B) compartilham a matriz entre embeddings e lm_head: aí não
        # dá para quantizar um e mover o outro para a CPU. O 7B tem as duas separadas.
        tied = bool(getattr(AutoConfig.from_pretrained(path), "tie_word_embeddings", False))
        kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
        # 4bit: [] quantiza também o lm_head (152k x 3584, ~1 GB em bf16).
        cfg = bnb_config(self.quant, None if tied or self.quant != "4bit" else [])
        if cfg is not None:
            kw["quantization_config"] = cfg
        model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
        if self.quant == "4bit" and not tied:
            embeddings_on_cpu(model)
        return Summarizer(model, tokenizer, stream=self.stream)
