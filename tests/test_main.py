"""python -m uocr chama a CLI do OCR."""
import runpy
import sys

import pytest


def test_python_m_uocr(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["uocr", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("uocr", run_name="__main__")
    assert e.value.code == 0 and "usage: uocr" in capsys.readouterr().out
