# Resumo por pasta

Depois do OCR de um `.zip`, o **Qwen2.5-7B-Instruct** pode resumir cada pasta do resultado. Cada pasta com `.md` ganha um `_resumo.md` com o resumo **dos arquivos dela**. Não há resumo geral.

```text
output/digitalizacoes/                output/digitalizacoes/
├─ cap1/pag1.md              ->       ├─ cap1/_resumo.md      (pag1 + pag2)
├─ cap1/pag2.md                       ├─ anexos/_resumo.md    (contrato)
├─ anexos/contrato.md                 └─ 2024/03/_resumo.md   (nota)
└─ 2024/03/nota.md                       (2024/ não tem .md próprio: sem resumo)
```

## Como usar

**CLI** (Git Bash, PowerShell, WSL ou Dev Container):

```bash
make resumo DIR=output/digitalizacoes
make resumo DIR=output/lote ARGS='--instrucao "foque em datas, valores e nomes"'
make resumo DIR=output/lote ARGS=--force                 # refaz as que já tinham resumo
make resumo DIR=output/lote ARGS="--max-input-tokens 3000"   # menos VRAM
```

O `DIR` aceita caminho do Windows (`C:/...`), como o `FILE` do `make ocr`. Sem `make`: `uocr-resumo output/lote` (no container ou na imagem) ou `bash scripts/run.sh uocr-resumo output/lote` (no WSL).

**Docker:** o resumo precisa da GPU livre, então pare a interface web antes:

```bash
make docker-down
make docker-resumo DIR=digitalizacoes    # pasta dentro de output/
```

**Web:** aba **Resumo**:
1. Escolha o lote na lista, que mostra as pastas de `output/` e `output/web/` com `.md`. Use **↻** para atualizar; um lote processado na aba *Lote (.zip)* entra na lista sozinho.
2. Opcional: escreva uma instrução adicional, marque **Refazer pastas já resumidas** ou ajuste os tokens.
3. Clique em **Resumir**. Os resumos aparecem na tela, e um `.zip` só com os `_resumo.md` fica disponível para baixar.

Na primeira vez, o modelo de OCR sai da GPU e o Qwen entra, o que leva ~1–2 min. O OCR só volta quando alguém usar as abas de OCR de novo.

## Opções

| Opção (CLI) | Web | Padrão | O que faz |
| --- | --- | --- | --- |
| `--instrucao` | Instrução adicional | — | Pedido extra incluído em todas as chamadas, ex.: "liste os valores em uma tabela" |
| `--force` | Refazer pastas já resumidas | — | Sem isso, pastas com `_resumo.md` mais novo que os `.md` são puladas |
| `--max-input-tokens` | Tokens de entrada | 4000 | Texto por chamada. Menos = menos VRAM e mais partes |
| `--max-new-tokens` | Tamanho máximo | 1024 | Limite de cada resumo final |
| `--quant` | `--summary-quant` (no `uocr-web`) | `4bit` | `8bit`/`bf16` precisam de GPU maior |
| `--model` | `--summary-model` (no `uocr-web`) | `Qwen/Qwen2.5-7B-Instruct` | Outro modelo de chat do Hugging Face (ou `UOCR_SUMMARY_MODEL`) |
| `--stream` | — | — | Mostra o texto enquanto é gerado |

## O `_resumo.md`

```markdown
# Resumo: cap1

<!-- gerado por uocr-resumo · Qwen/Qwen2.5-7B-Instruct -->
Arquivos (2): `pag1.md`, `pag2.md`

Visão geral em um parágrafo…

- ponto principal 1
- …
```

Os prompts pedem português do Brasil, um parágrafo de visão geral e os pontos principais em tópicos (nomes, datas, valores, decisões). Também pedem para não inventar nada e para ignorar ruído típico de OCR. Ficam em [`uocr/resumo/text.py`](../../uocr/resumo/text.py).

## Pastas grandes

Se o texto de uma pasta não cabe em `--max-input-tokens`:

1. Os arquivos são agrupados em partes. Um arquivo grande é partido por parágrafo.
2. Cada parte vira um resumo parcial curto, com a indicação de qual arquivo veio.
3. Os parciais são juntados num resumo final. Se nem eles couberem juntos, são juntados em grupos até sobrar um só.

Uma pasta com centenas de páginas funciona, mas leva várias chamadas ao modelo. Detalhes em [Arquitetura](Arquitetura.md#domínio-resumo).

## Memória (6 GB)

O 7B em bf16 ocupa ~15 GB. Para caber, o adapter:
- quantiza tudo em NF4, inclusive o `lm_head`;
- deixa a tabela de embeddings (~1 GB) na **CPU**.

Sobram ~2 GB para o texto. Se faltar memória, a pasta aparece com "sem memória na GPU" nos erros e as outras continuam. Reduza `--max-input-tokens`, por exemplo para 3000 ou 2500.

O resumo **não roda junto** com o OCR na mesma GPU de 6 GB:
- na web, a troca é automática;
- na CLI, não rode `make resumo` com a interface web aberta;
- no Docker, use `make docker-down` antes.
