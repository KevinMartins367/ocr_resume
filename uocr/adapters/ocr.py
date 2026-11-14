"""Adapter do Unlimited-OCR-2 para GPUs com pouca VRAM (ex.: 6 GB)."""
from __future__ import annotations

import os
import warnings

import torch

from .base import ModelAdapter, bnb_config

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast")

DEFAULT_MODEL = os.environ.get("UOCR_MODEL", "baidu/Unlimited-OCR-2")
QUANT_CHOICES = ("4bit", "8bit", "offload", "bf16")
DEFAULT_QUANT = os.environ.get("UOCR_QUANT", "4bit")

# Mantidos em bf16: encoder de visão, projetor, embeddings, lm_head e toda a atenção
# (DASH/roteador/kernels Triton). Só as MLPs/experts do MoE (~2,5B dos ~3,3B params) são quantizadas.
SKIP_MODULES = ["sam_model", "vision_model", "projector", "embed_tokens", "lm_head", "self_attn"]


def offload_device_map(path: str, offload_layers: int) -> dict:
    """Tudo na GPU, menos os experts das últimas `offload_layers` camadas MoE (vão p/ CPU).

    Cada expert vira uma entrada própria: o accelerate só move para a GPU os experts
    efetivamente chamados (6 de 64 por token).
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    lang = getattr(cfg, "language_config", None)
    get = (lambda k: lang.get(k) if isinstance(lang, dict) else getattr(lang, k, None)) if lang else (lambda k: None)
    n_layers = getattr(cfg, "num_hidden_layers", None) or get("num_hidden_layers")
    n_experts = getattr(cfg, "n_routed_experts", None) or get("n_routed_experts")
    first_dense = getattr(cfg, "first_k_dense_replace", None) or get("first_k_dense_replace") or 0
    moe_layers = list(range(first_dense, n_layers))[-offload_layers:] if offload_layers > 0 else []
    dmap = {"": 0}
    for li in moe_layers:
        for e in range(n_experts):
            dmap[f"model.layers.{li}.mlp.experts.{e}"] = "cpu"
    return dmap


class OcrAdapter(ModelAdapter):
    """`load()` devolve (model, tokenizer) do Unlimited-OCR-2."""

    kind = "OCR"
    quant_choices = QUANT_CHOICES

    def __init__(self, model_id: str = DEFAULT_MODEL, quant: str = DEFAULT_QUANT, offload_layers: int = 6):
        super().__init__(model_id, quant)
        self.offload_layers = offload_layers

    def _build(self, path: str):
        from transformers import AutoModel, AutoTokenizer

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        kw = dict(trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16,
                  low_cpu_mem_usage=True)
        if self.quant in ("4bit", "8bit"):
            kw["quantization_config"] = bnb_config(self.quant, SKIP_MODULES)
            kw["device_map"] = {"": 0}
        elif self.quant == "offload":
            kw["device_map"] = offload_device_map(path, self.offload_layers)
        model = AutoModel.from_pretrained(path, **kw).eval()
        if self.quant == "bf16":
            model = model.cuda()
        return model, tokenizer
