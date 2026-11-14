"""uocr.ocr.pdf: seleção de páginas, rasterização de PDF e leitura de imagens."""
import io

import pytest
from PIL import Image

from uocr.ocr.pdf import load_file, load_inputs, parse_pages, pdf_to_images


@pytest.mark.parametrize("spec, total, expected", [
    (None, 3, [0, 1, 2]),
    ("", 3, [0, 1, 2]),
    ("1-3,5", 5, [0, 1, 2, 4]),
    ("2-", 4, [1, 2, 3]),
    ("-2", 4, [0, 1]),
    (" 1, 1-2 ", 3, [0, 1]),       # espaços e duplicadas
    ("0,9", 3, []),                # fora do intervalo
    ("1,,3", 3, [0, 2]),           # vírgula sobrando
])
def test_parse_pages(spec, total, expected):
    assert parse_pages(spec, total) == expected


def test_pdf_to_images_from_bytes_respects_dpi(pdf_bytes):
    imgs = pdf_to_images(pdf_bytes, dpi=72)
    assert len(imgs) == 3
    assert imgs[0].size == (200, 300)
    assert pdf_to_images(pdf_bytes, dpi=144)[0].size == (400, 600)


def test_pdf_to_images_from_path_with_pages(tmp_path, pdf_bytes):
    p = tmp_path / "doc.pdf"
    p.write_bytes(pdf_bytes)
    assert len(pdf_to_images(p, dpi=72, pages="2")) == 1


def test_load_file_image_bytes_and_path(tmp_path, png_bytes):
    img, = load_file("a.png", png_bytes)
    assert (img.mode, img.size) == ("RGB", (50, 80))
    p = tmp_path / "a.png"
    p.write_bytes(png_bytes)
    assert load_file(p.name, p)[0].size == (50, 80)


def test_load_file_applies_exif_rotation(make_image):
    img = make_image(40, 20)
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: girar 90°
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert load_file("foto.jpg", buf.getvalue())[0].size == (20, 40)


def test_load_file_unsupported():
    with pytest.raises(ValueError, match="não suportado"):
        load_file("notas.txt", b"x")


def test_load_inputs_keeps_order(tmp_path, png_bytes, pdf_bytes):
    (tmp_path / "a.png").write_bytes(png_bytes)
    (tmp_path / "b.pdf").write_bytes(pdf_bytes)
    imgs = load_inputs([tmp_path / "a.png", tmp_path / "b.pdf"], dpi=72)
    assert [i.size for i in imgs] == [(50, 80)] + [(200, 300)] * 3


def test_load_inputs_empty_selection(tmp_path, pdf_bytes):
    (tmp_path / "b.pdf").write_bytes(pdf_bytes)
    with pytest.raises(ValueError, match="nenhuma página"):
        load_inputs([tmp_path / "b.pdf"], pages="9")


def test_image_is_converted_to_rgb():
    buf = io.BytesIO()
    Image.new("L", (10, 10)).save(buf, "PNG")
    assert load_file("g.png", buf.getvalue())[0].mode == "RGB"
