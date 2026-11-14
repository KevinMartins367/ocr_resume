"""uocr.ocr.pipeline: janelas de páginas, chamada ao modelo (mock) e arquivos gerados."""
import pytest

from conftest import FakeModel
from uocr.ocr.pipeline import _px, run_ocr


def test_px_scales_and_clamps():
    assert _px((0, 0, 1000, 1000), 200, 100) == (0, 0, 200, 100)
    assert _px((500, 500, 1500, 1500), 200, 100) == (100, 50, 200, 100)


def test_windows_and_model_arguments(tmp_path, fake_model, make_image, cuda_empty_cache):
    images = [make_image() for _ in range(5)]
    res = run_ocr(images, tmp_path, fake_model, "tok", batch_pages=2, tokens_per_page=100)

    assert [c["n"] for c in fake_model.calls] == [2, 2, 1]
    first, last = fake_model.calls[0], fake_model.calls[-1]
    assert first["max_new_tokens"] == 200 and last["max_new_tokens"] == 100
    assert first["ngram_window"] == 1024 and last["ngram_window"] == 128
    assert first["no_repeat_ngram_size"] == 35
    assert first["eval_mode"] is True and first["return_dict"] is True
    assert cuda_empty_cache.call_count == 3          # libera VRAM a cada janela
    assert (res.pages, res.tokens) == (5, 50)


@pytest.mark.parametrize("mode, crop", [("auto", None), ("gundam", True), ("base", False)])
def test_mode_maps_to_crop_mode(tmp_path, fake_model, make_image, mode, crop):
    run_ocr([make_image()], tmp_path, fake_model, "tok", mode=mode)
    assert fake_model.calls[0]["crop_mode"] is crop


def test_output_files(tmp_path, fake_model, make_image):
    res = run_ocr([make_image() for _ in range(3)], tmp_path, fake_model, "tok", batch_pages=2)

    md = (tmp_path / "result.md").read_text(encoding="utf-8")
    assert md == res.markdown
    for p in (1, 2, 3):
        assert f"Título {p}" in md and f"Texto da página {p} com acentuação." in md
        # figura: recortada (400x600 * box 0-1000) e referenciada no Markdown
        crop = tmp_path / "images" / f"page{p}_200_400_600_800.jpg"
        assert crop.exists()
        assert f"](images/page{p}_200_400_600_800.jpg)" in md
        assert (tmp_path / "boxes" / f"page_{p}.jpg").exists()
    raw = (tmp_path / "result_raw.md").read_text(encoding="utf-8")
    assert raw.count("<PAGE>") == 3 and "<BLOCK>" not in md
    assert [p.name for p in res.box_images] == ["page_1.jpg", "page_2.jpg", "page_3.jpg"]


def test_crop_size(tmp_path, fake_model, make_image):
    from PIL import Image
    run_ocr([make_image(400, 600)], tmp_path, fake_model, "tok")
    with Image.open(tmp_path / "images" / "page1_200_400_600_800.jpg") as crop:
        assert crop.size == (160, 240)


def test_model_debug_hidden_unless_stream(tmp_path, fake_model, make_image, capsys):
    run_ocr([make_image()], tmp_path, fake_model, "tok")
    assert "[DASH]" not in capsys.readouterr().out
    run_ocr([make_image()], tmp_path / "s", fake_model, "tok", stream=True)
    assert "[DASH]" in capsys.readouterr().out
    assert fake_model.calls[-1]["eval_mode"] is False


def test_warns_on_page_count_mismatch(tmp_path, make_image, capsys):
    run_ocr([make_image(), make_image()], tmp_path, FakeModel(pages_out=1), "tok")
    assert "[aviso]" in capsys.readouterr().err


def test_adds_missing_page_marker(tmp_path, make_image):
    run_ocr([make_image()], tmp_path, FakeModel(prefix_page=False), "tok")
    assert (tmp_path / "result_raw.md").read_text(encoding="utf-8").startswith("<PAGE>")


def test_progress_callback(tmp_path, fake_model, make_image):
    events = []
    run_ocr([make_image() for _ in range(3)], tmp_path, fake_model, "tok", batch_pages=2,
            progress=lambda i, n, msg: events.append((i, n, msg)))
    assert [(i, n) for i, n, _ in events] == [(0, 2), (1, 2), (2, 2)]
    assert events[-1][2] == "concluído" and "páginas 3-3" in events[1][2]


class _OddBlocksModel:
    """Blocos sem bbox, com bbox degenerada e de uma página que não existe: nada disso é desenhado."""

    def infer(self, tokenizer, image_file, **kw):
        text = ("<PAGE><BLOCK>text<FORK>sem caixa"
                "<BLOCK>image[500,500,500,600]<FORK>"
                "<BLOCK>text[100,100,900,200]<FORK>ok"
                "<PAGE><BLOCK>text[100,100,900,200]<FORK>página extra")
        return {"text": text, "token_ids": [1, 2], "num_pages_out": 2, "finish_reason": "stop"}


def test_odd_blocks_are_skipped_in_layout(tmp_path, make_image, capsys):
    res = run_ocr([make_image()], tmp_path, _OddBlocksModel(), "tok")
    assert "sem caixa" in res.markdown and "ok" in res.markdown
    assert not (tmp_path / "images").exists()                 # figura degenerada não é recortada
    assert [p.name for p in res.box_images] == ["page_1.jpg"]
