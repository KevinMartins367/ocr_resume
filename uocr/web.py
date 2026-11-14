"""Interface web (Gradio): uocr-web --quant 4bit --port 7860

Só interface: as abas chamam `services.ocr.OcrService` (Documento, Lote) e
`services.resumo.SummaryService` (Resumo). Os dois modelos não cabem juntos numa GPU de 6 GB:
um `adapters.base.GpuSlot` mantém um por vez e troca sob demanda; o lock da GPU fica nos serviços.
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

# Offline: sem telemetria/checagem de versão do Gradio (definido antes de importar o gradio).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr  # noqa: E402

from .adapters.base import GpuSlot  # noqa: E402
from .core.gpu import vram_report  # noqa: E402
from .adapters.ocr import DEFAULT_MODEL, DEFAULT_QUANT, QUANT_CHOICES, OcrAdapter  # noqa: E402
from .ocr.batch import BatchResult, make_zip  # noqa: E402
from .ocr.pdf import SUPPORTED_EXTS  # noqa: E402
from .adapters.resumo import DEFAULT_SUMMARY_MODEL, SUMMARY_QUANT_CHOICES, QwenAdapter  # noqa: E402
from .resumo.folders import SUMMARY_NAME, collect_summaries  # noqa: E402
from .resumo.text import DEFAULT_MAX_INPUT, DEFAULT_MAX_NEW  # noqa: E402
from .services.ocr import OcrOptions, OcrService  # noqa: E402
from .services.resumo import SummaryOptions, SummaryService  # noqa: E402

def _job_dir(out_root: Path, src: str) -> Path:
    return out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(src).stem}"


def _batch_summary(res: BatchResult, limit: int = 300) -> str:
    def block(title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        more = [f"- … e mais {len(items) - limit}"] if len(items) > limit else []
        return [f"### {title} ({len(items)})"] + [f"- `{x}`" for x in items[:limit]] + more + [""]

    return "\n".join(
        block("Gerados", res.done)
        + block("Erros", [f"{f}: {e}" for f, e in res.errors])
        + block("Ignorados", res.skipped))


def _progress_cb(progress):
    """Adapta o gr.Progress (ou None) para o callback (i, n, msg) do pipeline."""
    if progress is None:
        return None
    return lambda i, n, msg: progress((i, n), desc=msg)


# ---------------------------------------------------------------- OCR

def ocr_options(dpi, pages, batch_pages, mode, tokens_per_page) -> OcrOptions:
    """Valores da interface (sliders devolvem float, textbox vazio = todas) -> OcrOptions."""
    return OcrOptions(dpi=int(dpi), pages=pages or None, batch_pages=int(batch_pages), mode=mode,
                      tokens_per_page=int(tokens_per_page))


def handle_document(service: OcrService, out_root: Path, files, opts: OcrOptions, progress=None):
    """Aba Documento: arquivos enviados -> um documento. Retorna (md, md, galeria, zip, status)."""
    if not files:
        raise gr.Error("Envie ao menos um PDF ou imagem.")
    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        res = service.document(paths, _job_dir(out_root, paths[0]), opts, _progress_cb(progress))
    except ValueError as e:
        raise gr.Error(str(e))
    status = (f"{res.pages} página(s) · {res.tokens} tokens · {res.seconds:.0f}s · "
              f"{vram_report()} · saída: {res.out_dir}")
    return (res.markdown, res.markdown, [str(p) for p in res.box_images], str(make_zip(res.out_dir)),
            status)


def handle_zip(service: OcrService, out_root: Path, zfile, opts: OcrOptions, progress=None):
    """Aba Lote: .zip -> .zip com um .md por arquivo. Retorna (resumo, zip, status)."""
    if not zfile:
        raise gr.Error("Envie um arquivo .zip.")
    path = zfile if isinstance(zfile, str) else zfile.name
    if not zipfile.is_zipfile(path):
        raise gr.Error("O arquivo não é um .zip válido.")
    try:
        res = service.batch(path, _job_dir(out_root, path), opts, _progress_cb(progress))
    except ValueError as e:
        raise gr.Error(str(e))
    status = (f"{len(res.done)} .md gerado(s) · {len(res.errors)} erro(s) · "
              f"{len(res.skipped)} ignorado(s) · {res.seconds:.0f}s · saída: {res.out_dir}")
    return _batch_summary(res), str(make_zip(res.out_dir)), status


# ---------------------------------------------------------------- Resumo

def summary_base(out_root: Path) -> Path:
    """Onde procurar lotes: `output/` (pai de `output/web`), que tem os lotes da CLI e da web."""
    return out_root.parent if out_root.name == "web" else out_root


def list_batches(base: Path) -> list[str]:
    """Pastas com .md em `base` e em `base/web`, as mais novas primeiro (caminhos relativos)."""
    found = []
    for parent in (base, base / "web"):
        if not parent.is_dir():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d != base / "web" and any(
                    p.name != SUMMARY_NAME for p in d.rglob("*.md")):
                found.append(d)
    found.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.relative_to(base).as_posix() for d in found]


def _zip_summaries(folder: Path) -> Path:
    """`<lote>_resumos.zip` ao lado do lote, só com os `_resumo.md` (mesmas pastas)."""
    target = folder.with_name(f"{folder.name}_resumos.zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob(SUMMARY_NAME)):
            z.write(p, p.relative_to(folder).as_posix())
    return target


def handle_summary(service: SummaryService, base: Path, choice, opts: SummaryOptions, progress=None):
    """Aba Resumo: um `_resumo.md` por pasta do lote escolhido. Retorna (resumos, zip, status)."""
    if not choice:
        raise gr.Error("Escolha um lote (pasta gerada pelo OCR).")
    base = Path(base).resolve()
    folder = (base / choice).resolve()
    if not folder.is_relative_to(base) or not folder.is_dir():
        raise gr.Error(f"Pasta inválida: {choice}")
    try:
        res = service.summarize(folder, opts, _progress_cb(progress))
    except ValueError as e:
        raise gr.Error(str(e))
    text = collect_summaries(folder) or "_Nenhum resumo gerado._"
    if res.errors:
        text = "### Erros\n" + "\n".join(f"- `{f}`: {e}" for f, e in res.errors) + "\n\n---\n\n" + text
    status = (f"{len(res.done)} pasta(s) resumida(s) · {len(res.up_to_date)} já em dia · "
              f"{len(res.errors)} erro(s) · {res.seconds:.0f}s · saída: {folder}")
    return text, str(_zip_summaries(folder)), status


# ---------------------------------------------------------------- interface

def build_app(ocr: OcrService, summary: SummaryService, out_root: Path, info: str) -> gr.Blocks:
    base = summary_base(out_root)

    # Wrappers com `progress=gr.Progress()` na assinatura: é assim que o Gradio injeta a barra.
    def run_document(files, dpi, pages, batch_pages, mode, tpp, progress=gr.Progress()):
        return handle_document(ocr, out_root, files, ocr_options(dpi, pages, batch_pages, mode, tpp),
                               progress)

    def run_batch(zfile, dpi, pages, batch_pages, mode, tpp, progress=gr.Progress()):
        return handle_zip(ocr, out_root, zfile, ocr_options(dpi, pages, batch_pages, mode, tpp), progress)

    def run_summary(choice, instruction, force, max_in, max_new, progress=gr.Progress()):
        opts = SummaryOptions(max_input_tokens=int(max_in), max_new_tokens=int(max_new),
                              instruction=instruction or "", force=bool(force))
        return handle_summary(summary, base, choice, opts, progress)

    def refresh():
        return gr.Dropdown(choices=list_batches(base), value=None)

    with gr.Blocks(title="Unlimited-OCR") as app:
        gr.Markdown(f"## Unlimited-OCR 2 (local)\n{info}")
        with gr.Accordion("Opções do OCR", open=False):
            with gr.Row():
                dpi = gr.Slider(100, 300, value=200, step=10, label="DPI (PDF)")
                pages = gr.Textbox(label="Páginas do PDF (ex.: 1-3,5)", placeholder="todas")
                batch = gr.Slider(1, 10, value=4, step=1, label="Páginas por janela")
                mode = gr.Radio(["auto", "gundam", "base"], value="auto", label="Modo de visão")
                tpp = gr.Number(value=8000, precision=0, label="Máx. tokens por página")
        opts = [dpi, pages, batch, mode, tpp]

        with gr.Tab("Documento"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.File(label="PDFs / imagens (viram um documento)", file_count="multiple",
                                    file_types=sorted(SUPPORTED_EXTS))
                    run = gr.Button("Extrair", variant="primary")
                    status = gr.Markdown()
                    zip_out = gr.File(label="Resultado (.zip)")
                with gr.Column(scale=2):
                    with gr.Tab("Markdown"):
                        md_view = gr.Markdown()
                    with gr.Tab("Texto"):
                        md_raw = gr.Code(language="markdown")
                    with gr.Tab("Layout"):
                        gallery = gr.Gallery(columns=2, height="auto")
            run.click(run_document, [files] + opts, [md_view, md_raw, gallery, zip_out, status])

        with gr.Tab("Lote (.zip)"):
            gr.Markdown("Envie um `.zip` com pastas de imagens/PDFs. Cada arquivo vira um `.md` "
                        "no mesmo caminho (`cap1/pag1.jpg` → `cap1/pag1.md`), devolvido em outro `.zip`.")
            with gr.Row():
                with gr.Column(scale=1):
                    zip_in = gr.File(label="Arquivo .zip", file_count="single", file_types=[".zip"])
                    run_z = gr.Button("Processar lote", variant="primary")
                    zstatus = gr.Markdown()
                    zip_result = gr.File(label="Resultado (.zip com os .md)")
                with gr.Column(scale=2):
                    zsummary = gr.Markdown()

        with gr.Tab("Resumo"):
            gr.Markdown(
                "Gera um `_resumo.md` em **cada pasta** de um lote já processado, com o resumo dos "
                "`.md` daquela pasta (não há resumo geral). Usa o Qwen2.5-7B: na primeira vez, o "
                "modelo de OCR sai da GPU e o de resumo entra (~1–2 min); o OCR volta quando for "
                "usado de novo.")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        batch_dd = gr.Dropdown(choices=list_batches(base), label="Lote (pasta em output/)",
                                               scale=4)
                        refresh_btn = gr.Button("↻", scale=1, min_width=40)
                    instr = gr.Textbox(label="Instrução adicional (opcional)",
                                       placeholder="ex.: foque em datas, valores e nomes")
                    force = gr.Checkbox(label="Refazer pastas já resumidas")
                    with gr.Accordion("Opções do resumo", open=False):
                        max_in = gr.Slider(1000, 8000, value=DEFAULT_MAX_INPUT, step=500,
                                           label="Tokens de entrada por chamada (menos = menos VRAM)")
                        max_new = gr.Slider(256, 2048, value=DEFAULT_MAX_NEW, step=128,
                                            label="Tamanho máximo de cada resumo (tokens)")
                    run_s = gr.Button("Resumir", variant="primary")
                    sstatus = gr.Markdown()
                    szip = gr.File(label="Resumos (.zip só com os _resumo.md)")
                with gr.Column(scale=2):
                    sview = gr.Markdown()
            refresh_btn.click(refresh, None, batch_dd)
            run_s.click(run_summary, [batch_dd, instr, force, max_in, max_new], [sview, szip, sstatus])

        # Um lote novo aparece na lista da aba Resumo assim que termina.
        run_z.click(run_batch, [zip_in] + opts, [zsummary, zip_result, zstatus]).then(
            refresh, None, batch_dd)
    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="uocr-web")
    p.add_argument("--quant", choices=QUANT_CHOICES, default=DEFAULT_QUANT)
    p.add_argument("--offload-layers", type=int, default=6)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL, help="modelo da aba Resumo")
    p.add_argument("--summary-quant", choices=SUMMARY_QUANT_CHOICES, default="4bit")
    p.add_argument("--output", default="output/web")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args(argv)

    ocr_adapter = OcrAdapter(a.model, a.quant, a.offload_layers)
    qwen_adapter = QwenAdapter(a.summary_model, a.summary_quant)
    slot = GpuSlot(ocr_adapter, qwen_adapter)
    slot.get(ocr_adapter)   # o OCR já fica pronto; o resumo carrega na primeira vez que for pedido
    out_root = Path(a.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    base = summary_base(out_root)
    info = f"OCR `{a.model}` · `{a.quant}` | Resumo `{a.summary_model}` · `{a.summary_quant}`"
    app = build_app(OcrService(slot.provider(ocr_adapter)),
                    SummaryService(slot.provider(qwen_adapter), a.summary_model), out_root, info)
    app.queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, allowed_paths=[str(base)])


if __name__ == "__main__":
    main()
