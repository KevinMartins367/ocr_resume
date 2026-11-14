# Desenvolvimento

## Testes

São 131 testes em `tests/`, em pastas que espelham as de `uocr/`. Rodam em ~15 s **sem GPU e sem os modelos**, e cada camada é isolada com mocks. A cobertura é de **100%**, sem contar o `_postprocess.py`, que é cópia do upstream. O relatório por arquivo fica em [`docs/cobertura.md`](../cobertura.md), regravado a cada `make cobertura`.

| Pasta / arquivo | Testa | Mockado |
| --- | --- | --- |
| `adapters/test_base.py` | ciclo de vida do adapter, `GpuSlot` (um modelo por vez), `bnb_config` | adapter falso, GPU |
| `adapters/test_ocr.py` | 4bit/8bit/offload/bf16, carga pela pasta local, cache | `transformers` e a GPU |
| `adapters/test_resumo.py` | `lm_head` quantizado, embeddings na CPU (com um modelo minúsculo real), modelos com matriz compartilhada, `Summarizer` | `transformers`, tokenizer |
| `services/test_ocr.py` | validação antes de carregar o modelo, opções repassadas | pipeline e lote |
| `services/test_resumo.py` | validação antes de carregar, opções, **lock da GPU** entre threads | o LLM (`FakeLLM`) |
| `ocr/test_pdf.py` | seleção de páginas, PDF→imagem, EXIF | nada (usa PDFs/imagens gerados) |
| `ocr/test_pipeline.py` | janelas de páginas, argumentos do modelo, Markdown, recortes, blocos estranhos | o modelo (`FakeModel` devolve blocos no formato real) |
| `ocr/test_batch.py` | zip: caminhos maliciosos, acentos, espelhamento, erros, falta de VRAM | `run_ocr` |
| `ocr/test_cli.py` / `resumo/test_cli.py` | argumentos e roteamento das CLIs, código de saída | serviços e adapters |
| `resumo/test_text.py` | limpeza do Markdown, divisão no orçamento, map-reduce em várias rodadas | o LLM (`FakeLLM`) |
| `resumo/test_folders.py` | um resumo por pasta, sem resumo geral, pular os atualizados, `--force`, erros isolados | o LLM (`FakeLLM`) |
| `core/test_hub.py` | cache sem rede, download só quando falta, erro claro offline | `snapshot_download` |
| `core/test_gpu.py`, `core/test_util.py` | relatório de VRAM, ordem natural | `torch.cuda` |
| `test_web.py` | handlers das três abas, wrappers do Gradio, `main()`, segurança do caminho do lote | serviços |

Fixtures em [`tests/conftest.py`](../../tests/conftest.py):
- **`FakeModel`** imita o `model.infer(..., return_dict=True)` do OCR. Ele também imprime debug no stdout, como o modelo real, para testar que o pipeline o esconde.
- **`FakeLLM`** imita o `Summarizer`: conta palavras como tokens e devolve resumos numerados.
- **`hf`** mocka o `transformers` e simula a GPU e o cache local.

Os arquivos de teste com o mesmo nome em pastas diferentes (`test_cli.py`, `test_ocr.py`…) funcionam por causa do `--import-mode=importlib` no `pyproject.toml`.

```bash
make test                          # WSL, Linux ou Dev Container
make test ARGS="tests/ocr/test_batch.py -k safe"
make cobertura                     # + linhas não testadas (pytest-cov), salvo em docs/cobertura.md
make docker-test                   # no Docker
```

O `docker-test` monta `uocr/` e `tests/` da pasta (`PYTHONPATH=/app`), então testa o código atual sem precisar de rebuild.

### VS Code

- **Test Explorer:** só lista os testes quando o VS Code roda num ambiente com as dependências, ou seja, o Dev Container ou "WSL: Reopen Folder in WSL". No Python do Windows o erro é `No module named 'PIL'`.
- **Depuração** ([`.vscode/launch.json`](../../.vscode/launch.json), dentro do Dev Container): *Web: interface* e *CLI: OCR do arquivo aberto no editor*. As duas perguntam a quantização.
- **Pytest Runner** (fora do container): botões 💻 (WSL) e 🐳 (Docker) na barra de status.

## Dependências (uv)

As dependências ficam no [`pyproject.toml`](../../pyproject.toml), e as versões exatas no `uv.lock`. As do modelo (torch, transformers, tokenizers…) estão fixadas nas versões testadas pelo projeto oficial: **não atualize essas sem testar com o modelo real**. O lock só resolve Linux (`environments = ["sys_platform == 'linux'"]`).

Dentro do WSL, na pasta do projeto, com `export UV_PROJECT_ENVIRONMENT=~/.venvs/uocr` (o `scripts/run.sh` já faz isso):

```bash
uv add <pacote>         # adiciona dependência (atualiza pyproject + lock)
uv lock --upgrade       # atualiza versões
uv sync                 # aplica o lock no ambiente
```

Depois de mudar dependências, rode `make docker-build`: a camada de dependências da imagem só é refeita quando o `uv.lock` muda.

## Imagens Docker

| Imagem | Origem | Uso |
| --- | --- | --- |
| `uocr:latest` | estágio `app` | `make docker-up`, `docker-ocr` |
| `uocr:test` | estágio `test` | `make docker-test` |
| `vsc-unlimited-ocr-…` | estágio `dev` | Dev Container |

Volume `unlimited-ocr_uocr-data` (montado em `/data`):
- `/data/hf`: o modelo;
- `/data/triton`: o cache dos kernels compilados.

O nome fixo (`name: unlimited-ocr` no [`compose.yaml`](../../compose.yaml)) permite que o Dev Container reaproveite esse volume.

### Imagens `<none>`

Reconstruir uma imagem com a mesma tag deixa a anterior sem nome (`<none>`, "dangling"). Para elas não se acumularem:

- todas as imagens do projeto têm o rótulo `com.unlimited-ocr.project=uocr` (`LABEL` no estágio `base`);
- `docker-build`, `docker-up` e `docker-test` rodam, depois do build, `docker image prune -f --filter label=com.unlimited-ocr.project=uocr`, que apaga só as órfãs do projeto;
- o Dev Container roda o mesmo comando no `initializeCommand`, antes de cada build;
- `make docker-clean` faz a limpeza na hora.

### Cache de build

O BuildKit guarda o **cache de build** separado das imagens, e ele não aparece em `docker images`. Lá ficam:
- as camadas de builds antigos: cada mudança no `uv.lock` deixa para trás uma camada de dependências de vários GB;
- o cache do `uv` (`--mount=type=cache` no Dockerfile).

`make docker-clean` também roda `docker builder prune -f --filter until=168h` (valor de `CACHE_AGE`):

| Comando | Efeito |
| --- | --- |
| `make docker-clean` | Apaga o cache sem uso há mais de 7 dias (`CACHE_AGE=168h`); os builds recentes continuam rápidos |
| `make docker-clean CACHE_AGE=24h` | Mais agressivo |
| `make docker-clean CACHE_AGE=0h` | Todo o cache sem uso; o próximo build baixa as dependências de novo |

O cache de build não aceita filtro por rótulo, então essa limpeza vale para **todos** os projetos da máquina. Por isso ela não roda sozinha a cada build: só pelo `docker-clean`. O comando termina com `docker system df`, que mostra quanto espaço sobrou. Outra opção é deixar o próprio Docker Desktop limitar o cache: em *Settings → Docker Engine*, `"builder": {"gc": {"enabled": true, "defaultKeepStorage": "20GB"}}`.

Imagens antigas, construídas antes do rótulo existir, não são reconhecidas. Apague-as uma vez com `docker image prune -f`, que remove **todas** as imagens `<none>` da máquina.

## Convenções

- **Código, comentários e mensagens em português.**
- `_postprocess.py` não se edita (ver [Arquitetura](Arquitetura.md#domínio-ocr)).
- Os finais de linha são fixados no `.gitattributes`: `.sh` e `.py` sempre LF, porque um `.sh` com CRLF quebra no WSL.
- `output/` e `input/` ficam fora do git e da imagem.
- **Modelo novo** = uma subclasse de `ModelAdapter` em `uocr/adapters/` que implementa só `_build(path)`. Validação, cache, carga offline e liberação da GPU vêm da base.
- **Tarefa nova** = um `ModelService` em `uocr/services/`, usado pela CLI e pela web. A lógica fica no domínio (`uocr/ocr/`, `uocr/resumo/`) e recebe o modelo como parâmetro, e por isso dá para testar tudo com mocks.
- `web.py` e as CLIs só convertem a entrada (argumentos, campos da tela) em opções e chamam o serviço.
