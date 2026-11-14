"""uocr.core.util."""
from uocr.core.util import natural_key, print_progress


def test_natural_key():
    assert sorted(["pag10", "Pag2", "pag1"], key=natural_key) == ["pag1", "Pag2", "pag10"]


def test_print_progress(capsys):
    print_progress(0, 3, "1/3: a.jpg")
    assert capsys.readouterr().out == "   1/3: a.jpg\n"
