"""Adapters: carregam cada modelo do Hugging Face numa GPU pequena.

`base` tem o que é comum (ModelAdapter, GpuSlot, bnb_config); `ocr` e `resumo` só dizem
como montar o seu modelo.
"""
