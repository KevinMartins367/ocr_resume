# Unlimited-OCR. Funciona no Git Bash, no PowerShell, no shell do WSL e no Dev Container.
#   Docker: make docker-up | docker-down | docker-logs | docker-ocr FILE=x.pdf | docker-resumo DIR=lote
#           | docker-baixar | docker-test | docker-clean
#   Local:  make [start] | stop | ocr FILE=x.pdf | resumo DIR=output/lote | setup | baixar | test
#   make help
#
# No Windows os alvos "Local" rodam dentro do WSL (wsl.exe). No Linux (shell do WSL,
# Dev Container) rodam direto. Na imagem Docker (/opt/venv) os comandos já estão no PATH.

DISTRO ?= Ubuntu-24.04
PORT   ?= 7860
QUANT  ?= 4bit
ARGS   ?=
FILE   ?=
DIR    ?=

# PORT/QUANT chegam ao compose.yaml como variáveis de ambiente.
export PORT QUANT
# Impede o Git Bash (MSYS) de reescrever caminhos passados ao wsl.exe/docker.
export MSYS_NO_PATHCONV := 1

ifeq ($(OS),Windows_NT)
  WSL   = wsl -d $(DISTRO) --cd "$(CURDIR)" --exec
  RUN   = $(WSL) bash scripts/run.sh
  WHERE = WSL
else
  WSL   =
  # Na imagem do Docker/Dev Container o venv já está pronto e no PATH.
  IN_IMAGE := $(wildcard /opt/venv/bin/uocr)
  RUN   = $(if $(IN_IMAGE),,bash scripts/run.sh)
  WHERE = $(if $(IN_IMAGE),container,Linux)
endif

.DEFAULT_GOAL := start
.PHONY: start web stop ocr resumo setup baixar test cobertura help \
        docker-build docker-up docker-down docker-logs docker-ocr docker-resumo docker-baixar \
        docker-test docker-clean

# ---------------- Docker ----------------

# Reconstruir com a mesma tag deixa a imagem antiga como "<none>". Apaga só as do projeto
# (rótulo do Dockerfile); camadas em uso por outras imagens ficam.
PRUNE = docker image prune -f --filter label=com.unlimited-ocr.project=uocr

docker-build: ## Constrói a imagem
	docker compose build
	$(PRUNE)

docker-up: ## Sobe a interface web em segundo plano
	docker compose up -d --build
	$(PRUNE)
	@echo Subindo... acompanhe com: make docker-logs - depois abra http://localhost:$(PORT)

docker-down: ## Para e remove o container (o modelo continua no volume)
	docker compose down

docker-logs: ## Mostra os logs do serviço
	docker compose logs -f uocr

docker-ocr: ## OCR pela CLI: coloque o arquivo em input/ e use FILE=nome.pdf
	docker compose run --rm --no-deps uocr uocr /app/input/$(FILE) -o /app/output $(ARGS)

docker-resumo: ## Resumo por pasta de output/DIR (a GPU precisa estar livre: make docker-down antes)
	$(if $(DIR),,$(error Informe a pasta: make docker-resumo DIR=nome-do-lote))
	docker compose run --rm --no-deps uocr uocr-resumo "/app/output/$(DIR)" $(ARGS)

docker-baixar: ## Baixa os modelos de OCR e de resumo para o volume (depois funciona offline)
	docker compose run --rm --no-deps uocr python -m uocr.core.hub $(ARGS)

docker-test: ## Roda os testes dentro do Docker (código montado da pasta)
	docker compose --profile test build test
	$(PRUNE)
	docker compose run --rm test pytest $(ARGS)

# Cache de build (BuildKit): camadas de builds antigos e o cache do uv. Não aceita filtro por
# rótulo, então vale para qualquer projeto; só sai o que está sem uso há mais de CACHE_AGE.
# CACHE_AGE=0h apaga todo o cache sem uso (o próximo build baixa as dependências de novo).
CACHE_AGE ?= 168h

docker-clean: ## Apaga imagens "<none>" do projeto e o cache de build sem uso há CACHE_AGE (7 dias)
	$(PRUNE)
	docker builder prune -f --filter until=$(CACHE_AGE)
	docker system df

# ---------------- Local (WSL ou container) ----------------

start: ## Sobe a interface web
	@echo Carregando o modelo no $(WHERE)... depois abra http://localhost:$(PORT) - Ctrl+C para parar
	$(RUN) uocr-web --port $(PORT) --quant $(QUANT) $(ARGS)

web: start

ocr: ## OCR pela CLI: FILE=C:/docs/x.pdf (vários: FILE="a.jpg b.jpg"), saída em output/
	$(if $(FILE),,$(error Informe o arquivo: make ocr FILE=caminho/do/arquivo.pdf))
	$(RUN) uocr $(foreach f,$(FILE),"$(f)") --quant $(QUANT) $(ARGS)

resumo: ## Um _resumo.md por pasta: DIR=output/lote (aceita caminho do Windows)
	$(if $(DIR),,$(error Informe a pasta: make resumo DIR=output/nome-do-lote))
	$(RUN) uocr-resumo "$(DIR)" $(ARGS)

stop: ## Encerra a interface web, se ficou rodando
	-$(WSL) pkill -f uocr.web

setup: ## Instala o ambiente (WSL/Linux) e baixa os modelos (uma vez, com internet)
ifeq ($(WHERE),container)
	@echo Na imagem Docker o ambiente ja vem pronto - para uso offline rode: make baixar
else
	$(WSL) bash scripts/setup_wsl.sh
endif

baixar: ## Baixa os modelos de OCR e de resumo (com internet, uma vez; depois funciona offline)
	$(RUN) python -m uocr.core.hub $(ARGS)

test: ## Roda os testes (mocks, sem GPU)
	$(RUN) pytest $(ARGS)

cobertura: ## Testes com relatório de cobertura: no terminal e em docs/cobertura.md
	$(RUN) pytest --cov --cov-report=term-missing --cov-report=markdown:docs/cobertura.md $(ARGS)

help: ## Lista os comandos
	@echo Docker:
	@echo   make docker-up              - sobe a interface web, opcional: PORT=7860 QUANT=4bit
	@echo   make docker-down            - para o container
	@echo   make docker-logs            - logs do servico
	@echo   make docker-ocr FILE=x.pdf  - OCR de input/x.pdf para output/
	@echo   make docker-resumo DIR=lote - resumo por pasta de output/lote, com a GPU livre
	@echo   make docker-baixar          - baixa os modelos para uso offline
	@echo   make docker-test            - testes dentro do Docker
	@echo   make docker-clean           - apaga imagens sem nome e cache de build antigo, opcional: CACHE_AGE=168h
	@echo "Local ($(WHERE)):"
	@echo   make start          - sobe a interface web, opcional: PORT=7860 QUANT=4bit
	@echo   make stop           - encerra a interface web
	@echo   make ocr FILE=x.pdf - OCR de um arquivo, aceita caminho do Windows, saida em output/
	@echo   make resumo DIR=output/lote - um _resumo.md por pasta do lote
	@echo   make setup          - instala o ambiente e baixa os modelos
	@echo   make baixar         - so baixa os modelos, para uso offline
	@echo   make test           - roda os testes
	@echo   make cobertura      - testes com relatorio de cobertura, salvo em docs/cobertura.md
