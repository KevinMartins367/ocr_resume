"""uocr.resumo.cli: argumentos, uma chamada do serviço por pasta e código de saída."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from uocr.resumo import cli
from uocr.resumo.folders import SummaryResult


@pytest.fixture
def service(monkeypatch):
    svc = MagicMock()
    svc.summarize.side_effect = lambda folder, *a: SummaryResult(root=Path(folder), done=["cap1"])
    Service = MagicMock(return_value=svc)
    monkeypatch.setattr("uocr.services.resumo.SummaryService", Service)
    monkeypatch.setattr("uocr.adapters.resumo.QwenAdapter", MagicMock(name="QwenAdapter"))
    svc.Service = Service
    return svc


def test_parser_defaults():
    a = cli.build_parser().parse_args(["output/lote"])
    assert (a.quant, a.max_input_tokens, a.max_new_tokens, a.force) == ("4bit", 4000, 1024, False)
    assert a.model == "Qwen/Qwen2.5-7B-Instruct"


def test_missing_folder(service, tmp_path):
    with pytest.raises(SystemExit, match="não encontrada"):
        cli.main([str(tmp_path / "x")])
    service.Service.assert_not_called()


def test_each_folder_and_options(service, tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rc = cli.main([str(tmp_path / "a"), str(tmp_path / "b"), "--instrucao", "datas", "--force",
                   "--max-input-tokens", "3000"])
    assert rc == 0 and service.summarize.call_count == 2
    opts = service.summarize.call_args.args[1]
    assert (opts.instruction, opts.force, opts.max_input_tokens) == ("datas", True, 3000)
    assert "1 pasta(s) resumida(s)" in capsys.readouterr().out


def test_errors_set_exit_code(service, tmp_path):
    (tmp_path / "a").mkdir()
    service.summarize.side_effect = lambda folder, *a: SummaryResult(
        root=Path(folder), errors=[("cap1", "ruim")])
    assert cli.main([str(tmp_path / "a")]) == 1
    service.summarize.side_effect = ValueError("nenhum .md")
    assert cli.main([str(tmp_path / "a")]) == 1
