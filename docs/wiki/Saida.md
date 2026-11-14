# Saída

## Documento

Cada documento gera uma pasta em `output/<nome>/`. Para um exemplo real, veja a [foto de teste](Uso.md#exemplo-a-foto-de-teste).

```text
output/<nome>/
├─ result.md        # Markdown do documento
├─ result_raw.md    # saída crua do modelo
├─ images/          # figuras recortadas, referenciadas no result.md
└─ boxes/page_N.jpg # cada página com os blocos de layout desenhados
```

### `result.md`

- Títulos viram `#`…`######` conforme o nível detectado (`<LEVELn>`).
- Tabelas saem em **HTML** (`<table>`), o que preserva `colspan`/`rowspan`. Tabelas que continuam na página seguinte são unidas, e o cabeçalho repetido é removido.
- Parágrafos quebrados entre páginas são juntados.
- Figuras, gráficos e carimbos viram `![](images/pageN_x1_y1_x2_y2.jpg)`, com o recorte da página original.

### `result_raw.md`

É o que o modelo gerou, antes da conversão. Útil para depurar:

```text
<PAGE><BLOCK>title<LEVEL1>[100,50,900,100]<FORK>Capítulo 1
<BLOCK>text[100,150,900,300]<FORK>Primeiro parágrafo…
<BLOCK>image[200,400,600,800]<FORK>
<PAGE><BLOCK>append_table[...]<FORK><table>…</table>
```

As coordenadas vão de 0 a 1000, relativas à largura e à altura da página. Os prefixos `cross_*` e `append_*` marcam blocos que continuam em outra página.

### `boxes/`

Uma imagem por página com os blocos desenhados:

| Cor | Bloco |
| --- | --- |
| Vermelho (borda grossa) | título |
| Verde | tabela |
| Laranja | figura / gráfico |
| Azul | texto e o resto |

Use essas imagens para conferir se o modelo entendeu o layout. Se o texto saiu fora de ordem ou faltando, quase sempre dá para ver aqui o porquê.

## Lote (.zip)

Na CLI (`make ocr FILE=lote.zip`) ou na aba **Lote (.zip)** da web, cada imagem ou PDF vira um `.md` **no mesmo caminho**:

```text
entrada.zip                     output/entrada/
├─ cap1/pag1.jpg        ->      ├─ cap1/pag1.md
├─ cap1/pag2.jpg        ->      ├─ cap1/pag2.md  + pag2_images/ (se houver figuras)
├─ anexos/contrato.pdf  ->      ├─ anexos/contrato.md  (PDF inteiro = um .md)
└─ notas.txt                    └─ _relatorio.txt  (erros e arquivos ignorados)
```

Na web, a pasta volta compactada em outro `.zip`.

Depois do OCR, o [resumo](Resumo.md) pode acrescentar um `_resumo.md` em cada pasta com `.md`.

Regras:

- **Formatos:** png, jpg, jpeg, webp, bmp, tif, tiff e pdf. Outros arquivos vão para o relatório.
- **Ordem:** natural (`pag2` antes de `pag10`).
- **Leitura:** direto do zip, sem extrair para o disco.
- **Segurança:** entradas com `..`, caminho absoluto ou drive (`C:`) são rejeitadas. O limite é de 10 000 arquivos por zip e 512 MB por arquivo.
- **Lixo de sistema:** `__MACOSX`, arquivos ocultos, `Thumbs.db` e `desktop.ini` são ignorados sem aviso.
- **Zips do Windows:** nomes com acento gravados sem a flag UTF-8 (code page cp850) e o separador `\` são tratados.
- **Conflitos:** `foto.jpg` e `foto.png` na mesma pasta viram `foto.md` e `foto_png.md`.
- **Erros:** um arquivo com erro (inclusive falta de VRAM) vai para o `_relatorio.txt` e o lote continua. O relatório só é criado se houver erros ou arquivos ignorados.
- **Opções:** `--pages`, `--dpi`, `--mode` etc. valem para cada arquivo do lote.
