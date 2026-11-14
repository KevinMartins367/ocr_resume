"""uocr.resumo.text: limpeza do Markdown, divisão no orçamento de tokens e map-reduce."""
from conftest import FakeLLM
from uocr.resumo import text as t


def words(s: str) -> int:
    return len(s.split())


def test_clean_markdown():
    md = ("# Título\n\n![](images/page1_1_2_3_4.jpg)\n\n"
          "<table><tr><td>Item</td><td>Valor</td></tr><tr><td>Aluguel</td><td>1200</td></tr></table>")
    out = t.clean_markdown(md)
    assert "images/" not in out and "<" not in out
    assert "Item | Valor |" in out and "Aluguel | 1200 |" in out


def test_strip_fence():
    assert t.strip_fence("```markdown\n# A\ntexto\n```") == "# A\ntexto"
    assert t.strip_fence("```markdown\n# A\n```\n\n```markdown\n# B\n```") == "# A\n\n# B"
    assert t.strip_fence("# Sem cerca\n\n```py\nx\n```") == "# Sem cerca\n\n```py\nx\n```"


def test_pack_respects_budget():
    blocks = [" ".join(["a"] * 30), " ".join(["b"] * 30), " ".join(["c"] * 30)]
    chunks = t.pack(blocks, 70, words)
    assert len(chunks) == 2 and all(words(c) <= 70 for c in chunks)


def test_split_long_block_by_paragraph_line_and_chars():
    para = "\n\n".join(" ".join(["x"] * 20) for _ in range(5))       # 100 palavras em 5 parágrafos
    assert all(words(c) <= 45 for c in t.split_text(para, 45, words))
    one_line = " ".join(["y"] * 100)                                  # sem quebras: corta por caracteres
    pieces = t.split_text(one_line, 30, words)
    assert len(pieces) > 1 and "".join(pieces) == one_line


def test_fits_in_one_call(fake_llm):
    out = t.summarize_texts("cap1", [("a.md", "texto curto"), ("b.md", "outro")], fake_llm)
    assert out == "resumo 1" and len(fake_llm.calls) == 1
    user = fake_llm.calls[0]["user"]
    assert 'pasta "cap1"' in user and "## a.md" in user and "## b.md" in user
    assert fake_llm.calls[0]["max_new_tokens"] == t.DEFAULT_MAX_NEW


def test_map_reduce_when_too_big():
    llm = FakeLLM()
    docs = [(f"{i}.md", " ".join(["palavra"] * 300)) for i in range(6)]   # ~1800 palavras
    out = t.summarize_texts("lote", docs, llm, max_input_tokens=700, max_new_tokens=99,
                            instruction="foque em datas")
    parts = [c for c in llm.calls if "parte" in c["user"]]
    final = llm.calls[-1]
    assert len(parts) >= 3 and out == f"resumo {len(llm.calls)}"
    assert "resumos parciais" in final["user"] and final["max_new_tokens"] == 99
    assert all(c["max_new_tokens"] <= 512 for c in parts)                # parciais curtos
    assert all("foque em datas" in c["user"] for c in llm.calls)
    assert all(words(c["system"] + c["user"]) <= 700 for c in llm.calls)


class VerboseLLM(FakeLLM):
    """Resumos parciais longos: a primeira junção ainda não cabe e precisa de outra rodada."""

    def generate(self, system, user, max_new_tokens):
        super().generate(system, user, max_new_tokens)
        return " ".join(["detalhe"] * max_new_tokens)


def test_map_reduce_needs_several_rounds():
    llm = VerboseLLM()
    docs = [(f"{i}.md", " ".join(["palavra"] * 400)) for i in range(12)]   # ~4800 palavras
    t.summarize_texts("grande", docs, llm, max_input_tokens=700, max_new_tokens=50)
    combines = [c for c in llm.calls if "resumos parciais" in c["user"]]
    assert len(combines) >= 2                                   # junções intermediárias + final
    assert llm.calls[-1]["max_new_tokens"] == 50
    assert all(words(c["system"] + c["user"]) <= 700 for c in llm.calls)
