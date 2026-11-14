# Instalação

São três formas de rodar. Todas precisam de uma **GPU NVIDIA com ≥ 6 GB de VRAM** e de um driver recente no Windows.

| | Docker | WSL2 | Dev Container |
| --- | --- | --- | --- |
| Indicado para | uso no dia a dia | quem não quer Docker | desenvolver no VS Code |
| Instala no Windows | Docker Desktop + Make | WSL2 + Make | Docker Desktop + extensão Dev Containers |
| Onde ficam os modelos | volume `unlimited-ocr_uocr-data` | `~/.cache/huggingface` no WSL | o mesmo volume do Docker |
| Espaço | ~35 GB | ~28 GB | ~35 GB |

O **GNU Make** no Windows: `winget install ezwinports.make` ou `choco install make`. Ele funciona no Git Bash e no PowerShell.

São dois modelos: o de OCR (`baidu/Unlimited-OCR-2`, ~6,7 GB) e o de resumo (`Qwen/Qwen2.5-7B-Instruct`, ~15 GB). Internet só é necessária na instalação. Depois que os modelos estão baixados, tudo funciona **offline** (ver [Uso offline](#uso-offline)).

## Opção A: Docker (recomendado)

1. Instale o **Docker Desktop** com backend WSL2. O suporte a GPU NVIDIA já vem incluído.
2. Na pasta do projeto:

```bash
make docker-up      # constrói a imagem e sobe -> http://localhost:7860
make docker-baixar  # 1ª vez, com internet: baixa os dois modelos (~22 GB) para uso offline
make docker-logs    # acompanha a interface (carga do modelo, erros)
make docker-down    # para (o modelo continua salvo no volume)
```

Python, CUDA e gcc vêm todos dentro da imagem. Os modelos ficam em `/data`, que é o volume `uocr-data`, e por isso sobrevivem a `docker-down` e a rebuilds. Baixe os dois uma vez com `make docker-baixar`. Sem isso, cada modelo é baixado no primeiro uso, o que exige internet nesse momento.

## Opção B: WSL2 (sem Docker)

1. PowerShell como administrador: `wsl --install -d Ubuntu-24.04`.
2. Confira a GPU no WSL: `wsl -d Ubuntu-24.04 nvidia-smi`.
3. Na pasta do projeto (Git Bash ou PowerShell):

```bash
make setup   # 1ª vez: gcc, uv, Python 3.10, dependências e os dois modelos (~28 GB)
make         # sobe a interface web -> http://localhost:7860
```

O `make setup` roda o [`scripts/setup_wsl.sh`](../../scripts/setup_wsl.sh) dentro do Ubuntu. Esse script:

| Passo | Detalhe |
| --- | --- |
| Confere o `nvidia-smi` | sem ele, o driver do Windows precisa ser atualizado |
| Instala o `gcc` | o Triton compila os kernels em tempo de execução; pede a senha do `sudo` |
| Instala o [`uv`](https://docs.astral.sh/uv/) | gerenciador de pacotes Python |
| `uv sync --locked --python 3.10` | versões exatas do `uv.lock`, instaladas em `~/.venvs/uocr` |
| Testa o CUDA no torch | para logo aqui se a GPU não for vista |
| Baixa os modelos | `python -m uocr.core.hub` (OCR e resumo) |

O ambiente Python fica em `~/.venvs/uocr`, no disco do WSL e não em `/mnt/c` ou `/mnt/d`. No disco do Windows o torch ficaria muito lento.

**Outra distro:** veja o nome com `wsl -l -v` e passe `DISTRO`, ex.: `make setup DISTRO=Ubuntu`, `make DISTRO=Ubuntu`.

**Dentro do shell do WSL ou em Linux nativo:** o `make` detecta que não está no Windows e roda os comandos direto (`make setup`, `make`). Em Linux nativo isso não foi testado.

## Opção C: Dev Container (VS Code)

1. Instale a extensão **Dev Containers** e deixe o Docker Desktop aberto.
2. `Ctrl+Shift+P` → **Dev Containers: Reopen in Container**.

A imagem usa o estágio `dev` do [Dockerfile](../../Dockerfile), que tem as dependências, o pytest, o `git` e o `make`. O container recebe a GPU (`--gpus all`) e monta o mesmo volume do modelo usado pelo `make docker-up`. Lá dentro:

```bash
make                       # interface web
make ocr FILE=docs/exemplos/pagina-livro.jpg   # CLI, com a foto de teste
make test                  # testes
```

Sem GPU NVIDIA, remova `"--gpus", "all"` de [`.devcontainer/devcontainer.json`](../../.devcontainer/devcontainer.json). Os testes continuam funcionando.

## Uso offline

| Ambiente | Com internet, uma vez | Depois |
| --- | --- | --- |
| WSL2 | `make setup` (já baixa) ou `make baixar` | tudo offline |
| Docker | `make docker-up` (constrói a imagem) e `make docker-baixar` | tudo offline |
| Dev Container | build do container e `make baixar` | tudo offline |

Para conferir, desligue a rede e rode `make ocr FILE=docs/exemplos/pagina-livro.jpg`. O modelo carrega do cache sem tentar conectar. Se faltar um modelo, a mensagem diz para rodar `make baixar`.

O que **ainda precisa de internet**:
- construir ou reconstruir a imagem Docker;
- `make setup` e mudanças de dependência no `uv.lock`;
- trocar de modelo (`--model`, `--summary-model`) para um que ainda não foi baixado.
