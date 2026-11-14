"""uocr.core.gpu."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from uocr.core import gpu


def test_free(monkeypatch, cuda_empty_cache):
    collect = MagicMock()
    monkeypatch.setattr("gc.collect", collect)
    gpu.free()
    collect.assert_called_once()
    cuda_empty_cache.assert_called_once()


def test_vram_report_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert gpu.vram_report() == "sem CUDA"


def test_vram_report(monkeypatch):
    gb = 1024 ** 3
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda: 2 * gb)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 3 * gb)
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda i: SimpleNamespace(total_memory=6 * gb))
    assert gpu.vram_report() == "VRAM alocada 2.00 GB | pico 3.00 GB | total 6.00 GB"
