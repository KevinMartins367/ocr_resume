# Unlimited-OCR local

Wrapper local para o [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (modelo `baidu/Unlimited-OCR-2`). Converte PDFs, imagens e lotes `.zip` em Markdown. Depois, resume cada pasta de um lote com o **Qwen2.5-7B-Instruct**. Foi ajustado para GPUs com pouca VRAM (6 GB) e **funciona offline** depois que os modelos são baixados. Roda em Linux, via **Docker** ou **WSL2**, porque os kernels Triton do modelo de OCR exigem Linux.

O pós-processamento oficial (`utils/process_unlimitedOCR2.py`, licença MIT) foi copiado para `uocr/ocr/_postprocess.py`.

Documentação completa (uso, saída, arquitetura, desenvolvimento e solução de problemas) no **[wiki](docs/wiki/Home.md)**.

## Opção A: Docker (recomendado)

Pré-requisitos no Windows:

- **Docker Desktop** com backend WSL2. O suporte a GPU NVIDIA já vem incluído.
- **GPU NVIDIA** (≥ 6 GB de VRAM) com driver recente.
- **GNU Make**: `choco install make` ou `winget install ezwinports.make`.
- ~35 GB livres: imagem (~10 GB), modelo de OCR (~6,7 GB), modelo de resumo (~15 GB) e cache.

Python, CUDA, gcc e as dependências ficam todos dentro da imagem, então não é preciso instalar nada disso.

```bash
git clone <url-do-repositorio> Unlimited-OCR
cd Unlimited-OCR
make docker-up        # constrói a imagem e sobe -> http://localhost:7860
make docker-baixar    # 1ª vez, com internet: baixa os dois modelos (~22 GB) para uso offline
make docker-down      # para (o modelo fica salvo no volume "uocr-data")
```

Outros comandos:

```bash
make docker-up PORT=7861 QUANT=8bit       # porta / quantização diferentes
make docker-ocr FILE=contrato.pdf         # CLI: lê input/contrato.pdf, grava em output/
make docker-ocr FILE=lote.zip             # lote .zip pela CLI
make docker-down && make docker-resumo DIR=lote   # um _resumo.md por pasta de output/lote
make docker-test                          # testes (não precisam de GPU)
```

Cada rebuild deixa a imagem anterior sem nome (`<none>`). Os alvos `docker-*` e o Dev Container apagam essas sobras automaticamente, e `make docker-clean` faz isso na hora. O `docker-clean` também apaga o cache de build (camadas de builds antigos, cache do `uv`) sem uso há mais de 7 dias (`CACHE_AGE=168h`). Para ver quanto espaço o Docker ocupa: `docker system df`. Isso só vale para imagens construídas depois que o rótulo `com.unlimited-ocr.project` entrou no Dockerfile. As mais antigas saem uma vez com `docker image prune -f`, que apaga **todas** as imagens `<none>` da máquina.

Sem `make`: `docker compose up -d --build`, `docker compose down` e `docker compose run --rm uocr uocr /app/input/arquivo.pdf -o /app/output`.

## Opção B: WSL2 (sem Docker)

Pré-requisitos no Windows:

| Requisito | Como instalar / conferir |
| --- | --- |
| Windows 10/11 com **WSL2** e a distro **Ubuntu-24.04** | `wsl --install -d Ubuntu-24.04` (PowerShell como admin) |
| **GPU NVIDIA** (≥ 6 GB de VRAM) com driver recente no Windows | O driver do Windows já habilita CUDA no WSL. Confira com `wsl -d Ubuntu-24.04 nvidia-smi` |
| **GNU Make** | `choco install make` ou `winget install ezwinports.make` |
| Git Bash ou PowerShell | Qualquer um dos dois serve para rodar o `make` |
| ~28 GB livres no disco do WSL | Modelo de OCR (~6,7 GB) + modelo de resumo (~15 GB) + ambiente Python (~5 GB) |
| Internet na primeira instalação | Para baixar dependências e modelos. Depois, tudo funciona offline |

O Python **é necessário**, mas você não instala manualmente: o `make setup` usa o `uv` para baixar um Python 3.10 isolado, sem mexer no Python do sistema. O CUDA vem pelo driver do Windows e pelas bibliotecas do torch. Tudo abaixo é instalado automaticamente dentro do Ubuntu:

| Item | Para quê |
| --- | --- |
| `gcc` | O Triton compila os kernels do modelo em tempo de execução. Pede a senha do `sudo` |
| [`uv`](https://docs.astral.sh/uv/) | Gerenciador de pacotes Python (o "npm" do projeto) |
| Python 3.10 | Baixado pelo `uv` |
| Dependências do [pyproject.toml](pyproject.toml) | Versões exatas do `uv.lock` (torch 2.13, transformers 4.52.1, bitsandbytes, gradio…), instaladas em `~/.venvs/uocr` |
| Modelos `baidu/Unlimited-OCR-2` e `Qwen/Qwen2.5-7B-Instruct` | Baixados em `~/.cache/huggingface` |

### Começando (WSL2)

No Git Bash ou no PowerShell:

```bash
git clone <url-do-repositorio> Unlimited-OCR
cd Unlimited-OCR
make setup    # 1ª vez: instala tudo no WSL e baixa os modelos (demora; ~28 GB)
make          # sobe a interface web -> http://localhost:7860 (Ctrl+C para parar)
make test     # testes (não precisam de GPU)
```

Se a sua distro WSL tiver outro nome (veja com `wsl -l -v`), passe `DISTRO`, ex.: `make setup DISTRO=Ubuntu` e `make DISTRO=Ubuntu`.

No shell do WSL ou em Linux nativo, o `make` roda os comandos direto, sem passar pelo `wsl.exe`: `make setup` e depois `make`. Em Linux nativo isso não foi testado.

Outros comandos:

```bash
make start PORT=7861 QUANT=8bit   # porta / quantização diferentes
make stop                         # encerra, se o serviço ficou rodando no WSL
make help
```

## Testes

São 131 testes em `tests/`, que espelham as pastas de `uocr/` e rodam em ~15 s **sem GPU e sem os modelos**. Cada camada é isolada com mocks. A cobertura é de **100%**, sem contar o `_postprocess.py`, que é cópia do projeto oficial.

```bash
make test          # WSL, Linux ou Dev Container
make cobertura     # testes + linhas não cobertas (salvo em docs/cobertura.md)
make docker-test   # no Docker
```

O `docker-test` monta `uocr/` e `tests/` da pasta, então testa o código atual sem rebuild. A tabela do que cada arquivo testa está no [wiki](docs/wiki/Desenvolvimento.md#testes).

### VS Code

O Test Explorer só lista os testes se o VS Code estiver rodando no ambiente que tem as dependências. No Python do Windows dá `No module named 'PIL'`. Use o **Dev Container** ([.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)):

1. Instale a extensão **Dev Containers** e deixe o Docker Desktop aberto.
2. `Ctrl+Shift+P` → **"Dev Containers: Reopen in Container"**. O primeiro build demora; os seguintes reaproveitam o cache.
3. Abra o painel **Testing** (ícone de frasco). Os testes aparecem.

Dentro do container, o terminal também tem GPU e o modelo baixado. Os mesmos comandos funcionam lá: `make`, `make ocr FILE=arquivo.pdf`, `make resumo DIR=output/lote` e `make test`. Também dá para chamar `uocr-web`, `uocr` e `uocr-resumo` direto.

Fora do container, a extensão **Pytest Runner** roda testes pelos botões 💻 (WSL) e 🐳 (Docker) na barra de status.

## Dependências (uv)

As dependências ficam no [pyproject.toml](pyproject.toml), e as versões exatas no `uv.lock`. Os comandos abaixo rodam dentro do WSL, na pasta do projeto, com `export UV_PROJECT_ENVIRONMENT=~/.venvs/uocr` (o `scripts/run.sh` já faz isso):

```bash
uv add <pacote>         # adiciona dependência (atualiza pyproject + lock)
uv lock --upgrade       # atualiza versões
uv sync                 # aplica o lock no ambiente
uv run uocr-web         # executa um comando do projeto
```

## Uso (CLI)

No Git Bash ou no PowerShell, na pasta do projeto. O arquivo pode estar em qualquer lugar do Windows. O resultado vai para `output/`:

```bash
make ocr FILE=C:/docs/contrato.pdf                       # gera output/contrato/result.md
make ocr FILE=docs/exemplos/pagina-livro.jpg ARGS="--mode gundam"   # foto de exemplo, com tiles (padrão p/ 1 página)
make ocr FILE=livro.pdf ARGS="--pages 1-20 --batch-pages 4"
make ocr FILE="p1.jpg p2.jpg p3.jpg" ARGS=--combine      # várias imagens = um documento
make ocr FILE=digitalizacoes.zip                         # lote: um .md por arquivo, mesmas pastas
make ocr FILE=x.pdf QUANT=8bit                           # outra quantização
```

Caminhos relativos partem da raiz do projeto. Caminhos do Windows (`C:\...` ou `C:/...`) são convertidos para `/mnt/c/...` pelo `scripts/run.sh`. No Git Bash, use `/` ou ponha o caminho entre aspas, porque o bash remove as `\`. Caminhos com espaços não funcionam no `FILE`: renomeie ou copie o arquivo para a pasta do projeto.

Dentro do WSL também funciona: `bash scripts/run.sh uocr arquivo.pdf` ou `bash scripts/run.sh uocr-web`.

### Opções

| Opção | Padrão | O que faz |
| --- | --- | --- |
| `--quant` | `4bit` | `4bit`/`8bit` quantizam só os experts do MoE (visão e atenção continuam em bf16); `offload` usa bf16 com parte dos experts na CPU; `bf16` só serve para GPU ≥ 10 GB |
| `--batch-pages` | 4 | Páginas por chamada ao modelo. Mais páginas dão mais contexto entre páginas e usam mais VRAM |
| `--mode` | `auto` | `gundam` divide a página em tiles; `base` usa uma visão global por página. `auto` usa gundam para 1 página e base para várias |
| `--dpi` | 200 | Resolução usada para rasterizar PDFs |
| `--pages` | todas | Seleção de páginas, ex.: `1-3,5` |
| `--tokens-per-page` | 8000 | Limite de geração por página |
| `--stream` | — | Mostra o texto enquanto é gerado (com os logs de debug do modelo) |

Variáveis de ambiente: `UOCR_QUANT`, `UOCR_MODEL`, `UOCR_SUMMARY_MODEL` e `UOCR_VENV`. A distro do WSL usada pelo `make` vem de `DISTRO` (padrão `Ubuntu-24.04`).

### Desempenho medido (RTX 3060 Laptop, 6 GB)

Foto de página de livro em pt-br, 772 tokens gerados:

| Modo | Tempo | Pico de VRAM | Texto |
| --- | --- | --- | --- |
| `4bit` | ~85 s | 4,0 GB | igual ao 8bit |
| `8bit` | ~254 s | 5,2 GB | referência |

A primeira execução de cada processo é um pouco mais lenta porque o Triton compila os kernels.

## Saída

```text
output/<nome>/
├─ result.md        # Markdown do documento (títulos, tabelas HTML, figuras; junta blocos entre páginas)
├─ result_raw.md    # saída crua do modelo: <PAGE><BLOCK>label<LEVELn>[x1,y1,x2,y2]<FORK>texto
├─ images/          # figuras recortadas, referenciadas no result.md
└─ boxes/page_N.jpg # página com os blocos de layout desenhados
```

### Lote (.zip)

Na CLI (`uocr arquivo.zip`) ou na aba **Lote (.zip)** da web, cada imagem ou PDF do zip vira um `.md` no mesmo caminho. Na web, o resultado volta como outro `.zip`:

```text
entrada.zip                     output/entrada/
├─ cap1/pag1.jpg        ->      ├─ cap1/pag1.md
├─ cap1/pag2.jpg        ->      ├─ cap1/pag2.md  + pag2_images/ (se houver figuras)
├─ anexos/contrato.pdf  ->      ├─ anexos/contrato.md  (PDF inteiro = um .md)
└─ notas.txt                    └─ _relatorio.txt  (erros e arquivos ignorados)
```

Regras do lote:

- **Formatos**: png, jpg, jpeg, webp, bmp, tif, tiff e pdf. Outros arquivos vão para o relatório.
- **Leitura**: os arquivos são lidos direto do zip, sem extrair para o disco.
- **Caminhos inválidos**: entradas com `..` ou drive são rejeitadas.
- **Lixo de sistema**: `__MACOSX` e arquivos ocultos são ignorados.
- **Nomes do Windows**: acentos e o separador `\` de zips criados no Windows são tratados.
- **Erros**: um arquivo com erro não interrompe o lote.

## Resumo por pasta (Qwen2.5-7B)

Depois do OCR de um `.zip`, cada pasta do resultado pode ganhar um `_resumo.md` com o resumo dos `.md` **daquela pasta**. Não há resumo geral: uma pasta-mãe sem `.md` próprios não ganha resumo.

```bash
make resumo DIR=output/digitalizacoes                       # um _resumo.md por pasta
make resumo DIR=output/lote ARGS='--instrucao "foque em datas e valores"'
make resumo DIR=output/lote ARGS=--force                    # refaz as que já tinham resumo
```

Na web, use a aba **Resumo** e escolha o lote. Lá, o modelo de OCR sai da GPU e o de resumo entra (~1–2 min); o OCR volta quando for usado de novo. Os dois modelos não cabem juntos em 6 GB.

```text
output/lote/                     output/lote/
├─ cap1/pag1.md          ->      ├─ cap1/_resumo.md     (pag1 + pag2)
├─ cap1/pag2.md                  └─ anexos/_resumo.md   (contrato)
└─ anexos/contrato.md
```

- **Pastas grandes** são resumidas em partes. Os resumos parciais são juntados no final.
- **Pastas que já têm `_resumo.md`** mais novo que os `.md` são puladas; use `--force` para refazer.
- **Em 6 GB**, o Qwen roda em 4 bits com os embeddings na CPU. Se faltar memória, reduza `--max-input-tokens` (padrão 4000).

## Offline

Com internet, uma vez: `make setup` (WSL, já baixa os modelos), `make baixar` ou `make docker-baixar`. Depois disso, OCR, resumo e interface web funcionam **sem internet**. Os modelos são carregados direto do cache, sem consultar o Hugging Face, e o Gradio roda sem telemetria. Sem internet e sem o modelo no cache, o erro diz qual comando rodar.
