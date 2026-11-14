"""uocr.core.hub: carregar do cache sem rede; baixar só quando falta e há internet."""
from unittest.mock import MagicMock

import pytest

from uocr.core import hub


@pytest.fixture
def snap(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return mock


def test_local_dir_is_used_as_is(tmp_path, snap):
    assert hub.local_model_path(str(tmp_path)) == str(tmp_path)
    snap.assert_not_called()


def test_cached_model_never_touches_network(snap):
    snap.return_value = "/cache/snap"
    assert hub.local_model_path("org/modelo") == "/cache/snap"
    snap.assert_called_once_with("org/modelo", local_files_only=True)


def test_missing_and_online_downloads(snap):
    snap.side_effect = [FileNotFoundError("sem cache"), "/cache/novo"]
    assert hub.local_model_path("org/modelo") == "/cache/novo"
    assert snap.call_args.kwargs == {"ignore_patterns": hub.IGNORE}


def test_missing_and_offline_explains(snap, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snap.side_effect = FileNotFoundError("sem cache")
    with pytest.raises(RuntimeError, match="make baixar"):
        hub.local_model_path("org/modelo")
    assert snap.call_count == 1                         # nem tenta a rede


def test_missing_and_no_network_explains(snap):
    snap.side_effect = [FileNotFoundError("sem cache"), OSError("sem rede")]
    with pytest.raises(RuntimeError, match="make baixar"):
        hub.local_model_path("org/modelo")


@pytest.mark.parametrize("argv, n", [([], 2), (["--so-ocr"], 1)])
def test_main_downloads(snap, argv, n):
    snap.return_value = "/cache/x"
    assert hub.main(argv) == 0
    assert snap.call_count == n
