"""Texto do resumo: limpeza do Markdown do OCR, divisão em partes que cabem no contexto,
prompts e o map-reduce que transforma os .md de uma pasta em um resumo."""
from __future__ import annotations

import re
from typing import Callable, Protocol

DEFAULT_MAX_INPUT = 4000    # tokens de entrada por chamada (limita a VRAM do contexto)
DEFAULT_MAX_NEW = 1024      # tokens do resumo final de cada pasta

SYSTEM = (
    "Você resume documentos digitalizados por OCR. O texto pode ter erros de reconhecimento: "
    "corrija o que for óbvio e ignore ruído como números de página e cabeçalhos repetidos. "
    "Escreva sempre em português do Brasil, em Markdown, sem inventar nada que não esteja no texto."
)
PROMPT_FINAL = (
    'Resuma o conteúdo da pasta "{folder}", formada pelos arquivos abaixo. Comece com um '
    "parágrafo de visão geral e depois liste os pontos principais em tópicos (nomes, datas, "
    "valores e decisões, quando houver). Se os arquivos tratarem de assuntos diferentes, "
    "faça um tópico por arquivo.{extra}\n\n{text}"
)
PROMPT_PART = (
    'O texto abaixo é a parte {i} de {n} da pasta "{folder}". Resuma em tópicos curtos tudo o '
    "que for importante (fatos, nomes, datas, valores), indicando de qual arquivo veio.{extra}"
    "\n\n{text}"
)
PROMPT_COMBINE = (
    'Abaixo estão resumos parciais da pasta "{folder}". Junte-os em um único resumo: um '
    "parágrafo de visão geral e depois os pontos principais em tópicos, sem repetir "
    "informações.{extra}\n\n{text}"
)


class TextModel(Protocol):
    """O que o resumo precisa de um modelo (o real é `adapters.resumo.Summarizer`)."""
    def count(self, text: str) -> int: ...
    def generate(self, system: str, user: str, max_new_tokens: int) -> str: ...


_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CELL_END_RE = re.compile(r"\s*</t[dh]>\s*", re.I)
_ROW_END_RE = re.compile(r"\s*</tr>\s*", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_FENCE_LINE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.M)


def clean_markdown(md: str) -> str:
    """Tira o que só gasta tokens: links de figuras e tags HTML das tabelas (vira `a | b |`)."""
    text = _IMG_RE.sub("", md)
    text = _CELL_END_RE.sub(" | ", text)
    text = _ROW_END_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_fence(text: str) -> str:
    """O Qwen às vezes embrulha a resposta em ```markdown ... ``` (às vezes em vários blocos).
    Se a resposta começa assim, tira as linhas de cerca e fica com o Markdown."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    return re.sub(r"\n{3,}", "\n\n", _FENCE_LINE_RE.sub("", text)).strip()


def split_text(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Divide `text` em pedaços de até `budget` tokens: por parágrafo, depois por linha
    e, em último caso, por caracteres."""
    if count(text) <= budget:
        return [text]
    for sep in ("\n\n", "\n"):
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) > 1:
            return pack(parts, budget, count, sep)
    step = max(1, int(len(text) * budget / count(text) * 0.9))
    return [text[i:i + step] for i in range(0, len(text), step)]


def pack(blocks: list[str], budget: int, count: Callable[[str], int], sep: str = "\n\n") -> list[str]:
    """Agrupa blocos em pedaços de até `budget` tokens, sem partir um bloco que caiba."""
    chunks: list[str] = []
    cur: list[str] = []
    for block in blocks:
        for piece in split_text(block, budget, count):
            if cur and count(sep.join(cur + [piece])) > budget:
                chunks.append(sep.join(cur))
                cur = []
            cur.append(piece)
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def summarize_texts(folder: str, docs: list[tuple[str, str]], llm: TextModel, *,
                    max_input_tokens: int = DEFAULT_MAX_INPUT, max_new_tokens: int = DEFAULT_MAX_NEW,
                    instruction: str = "") -> str:
    """Resumo de uma pasta a partir de [(nome do arquivo, markdown)].

    Cabe no contexto: uma chamada. Não cabe: resume cada parte (resumos parciais curtos)
    e junta os parciais, repetindo a junção até sobrar um só (map-reduce)."""
    extra = f"\n\nInstrução adicional: {instruction.strip()}" if instruction.strip() else ""
    overhead = llm.count(SYSTEM + PROMPT_COMBINE.format(folder=folder, extra=extra, text="")) + 64
    budget = max(256, max_input_tokens - overhead)
    part_new = max(128, min(512, budget // 4))   # parciais curtos: cada junção encolhe o texto

    blocks = [f"## {name}\n\n{clean_markdown(md)}" for name, md in docs]
    chunks = pack(blocks, budget, llm.count)
    if len(chunks) == 1:
        return llm.generate(SYSTEM, PROMPT_FINAL.format(folder=folder, extra=extra, text=chunks[0]),
                            max_new_tokens)
    partials = [llm.generate(SYSTEM, PROMPT_PART.format(i=i, n=len(chunks), folder=folder,
                                                        extra=extra, text=c), part_new)
                for i, c in enumerate(chunks, 1)]
    while True:
        groups = pack(partials, budget, llm.count)
        if len(groups) == 1:
            return llm.generate(SYSTEM, PROMPT_COMBINE.format(folder=folder, extra=extra, text=groups[0]),
                                max_new_tokens)
        partials = [llm.generate(SYSTEM, PROMPT_COMBINE.format(folder=folder, extra=extra, text=g), part_new)
                    for g in groups]
