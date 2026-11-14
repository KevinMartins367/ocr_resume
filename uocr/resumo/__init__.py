"""Resumo por pasta com Qwen2.5-7B-Instruct: cada pasta com .md ganha um `_resumo.md`.

Não há resumo geral: as subpastas não entram no resumo da pasta-mãe.

    output/lote/                     output/lote/
    ├─ cap1/pag1.md          ->      ├─ cap1/_resumo.md   (pag1 + pag2)
    ├─ cap1/pag2.md                  └─ anexos/_resumo.md (contrato)
    └─ anexos/contrato.md

Módulos: `text` (limpeza, divisão, prompts e map-reduce) e `folders` (percorre as pastas
e grava os `_resumo.md`). O modelo fica em `adapters.resumo`; a tarefa, em `services.resumo`.
"""
