# Solução de problemas

## Instalação e ambiente

**`CUDA indisponível. Rode dentro do WSL2 com driver NVIDIA.`**
O torch não viu a GPU.
- **WSL:** confira com `wsl -d Ubuntu-24.04 nvidia-smi`. Se falhar, atualize o driver NVIDIA **do Windows**; não instale driver dentro do WSL.
- **Docker:** confira se o Docker Desktop usa o backend WSL2 e teste com `docker run --rm --gpus all ubuntu nvidia-smi`.

**`nvidia-smi não encontrado no WSL`** (no `make setup`)
É a mesma causa: driver do Windows desatualizado.

**`wsl: distribuição não encontrada` / o make não acha a distro**
Veja o nome com `wsl -l -v` e passe `DISTRO=<nome>` em todos os comandos.

**`make: command not found`**
- **Windows:** `winget install ezwinports.make`, depois abra um terminal novo.
- **Dev Container:** reconstrua o container (*Dev Containers: Rebuild Container*); o estágio `dev` já inclui o `make`.

**Test Explorer vazio / `No module named 'PIL'`**
O VS Code está usando o Python do Windows. Abra o projeto no Dev Container ou no WSL (ver [Desenvolvimento](Desenvolvimento.md#vs-code)).

## Execução

**`arquivo não encontrado: ...`**
- Para isolar o problema, rode primeiro com a foto de exemplo: `make ocr FILE=docs/exemplos/pagina-livro.jpg`.
- No Git Bash, `C:\docs\x.pdf` sem aspas perde as `\`: use `C:/docs/x.pdf`.
- Caminhos com espaços não funcionam no `FILE`.
- Caminhos relativos partem da raiz do projeto.
- No `docker-ocr`, o arquivo precisa estar em `input/`.

**Sem memória na GPU (`OutOfMemoryError` / "sem memória na GPU" no relatório)**, em ordem de impacto:
1. `QUANT=4bit` (é o padrão; confira se não está em `8bit`/`bf16`).
2. `ARGS="--batch-pages 1"`: menos páginas por chamada.
3. `ARGS="--dpi 150"` para PDFs.
4. `QUANT=offload` com mais camadas na CPU: `ARGS="--offload-layers 10"`.
5. Feche outros programas que usam a GPU (navegador com aceleração, jogos).

**A primeira página demora muito**
O Triton compila os kernels na primeira execução de cada processo. No Docker, o cache fica em `/data/triton` e sobrevive a reinícios.

**`[aviso] janela N: X página(s) enviadas, Y retornadas (length)`**
O modelo atingiu o limite de tokens antes de terminar a janela, e o fim do texto foi cortado. Aumente `--tokens-per-page` ou diminua `--batch-pages`.

**Texto fora de ordem, faltando ou repetido**
- Abra `boxes/page_N.jpg` para ver como o modelo dividiu a página.
- Para uma página só, `--mode gundam` (tiles) costuma ler melhor letras pequenas.
- Fotos tortas ou com baixa resolução pioram a leitura; o EXIF já é corrigido.

**A interface web não abre em `localhost:7860`**
- Aguarde a mensagem do Gradio: o modelo leva um tempo para carregar, e no Docker acompanhe com `make docker-logs`.
- Porta ocupada: use `PORT=7861`.
- Se ficou um processo antigo no WSL: `make stop`.

## Resumo e offline

**`o modelo ... não está baixado e o modo offline está ativo` / `... não foi possível baixar`**
O modelo não está no cache e não há internet. Com internet, rode uma vez `make baixar` (ou `make docker-baixar`). Depois tudo funciona offline.

**"sem memória na GPU" no resumo de uma pasta**
1. Confira se não há outro processo usando a GPU: a interface web aberta (`make stop` / `make docker-down`) ou outro `make ocr`.
2. Reduza o texto por chamada: `ARGS="--max-input-tokens 3000"` (ou 2500).
3. Feche programas que usam a GPU no Windows.

**O resumo demora para começar na web**
Na primeira vez, o modelo de OCR sai da GPU e o Qwen entra, o que leva ~1–2 min. O mesmo acontece ao voltar para o OCR. Fazer vários resumos em seguida não troca de novo.

**Uma pasta não ganhou `_resumo.md`**
- Ela não tem `.md` próprios, só subpastas: não há resumo geral, por decisão de projeto.
- Ela já tinha um `_resumo.md` mais novo que os `.md` e foi pulada: use `--force` (ou marque *Refazer* na web).
- Ela deu erro: veja a lista de erros no terminal ou no topo da aba Resumo.

**O resumo veio dentro de um bloco de código**
O código remove a cerca ` ``` ` quando a resposta inteira começa com ela. Se aparecer no meio do texto, foi o modelo que escreveu assim; refaça com `--force` e uma `--instrucao` pedindo texto corrido.

## Docker

**Várias imagens `<none>` no Docker Desktop**
São versões antigas deixadas pelos rebuilds. Hoje elas são limpas automaticamente (ver [Desenvolvimento](Desenvolvimento.md#imagens-none)). As antigas, de antes do rótulo, saem uma vez com `docker image prune -f`.

**O modelo é baixado de novo a cada `docker-up`**
Não deveria: ele fica no volume `unlimited-ocr_uocr-data`. Confira se o volume existe com `docker volume ls`, e não use `docker compose down -v`, porque o `-v` apaga os volumes.

**Pouco espaço em disco**
- `docker system df` mostra o que ocupa espaço: imagens, containers, volumes e cache de build.
- `make docker-clean` apaga as imagens órfãs do projeto e o cache de build sem uso há 7 dias; `CACHE_AGE=0h` apaga todo o cache sem uso.
- O modelo (~6,7 GB) fica no volume. Só apague o volume se quiser baixar tudo de novo.
