import json

from fireworkshack.cli import main


def test_offline_cli_and_default_run(capsys):
    assert main(["preflight", "--offline"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["network_calls"] == 0
    assert main(["run", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["network_calls"] == 0
    assert report["mode"] == "dry-run"


def test_paid_execution_requires_matching_approval(capsys):
    assert main(["run", "--execute"]) == 2
    assert "approval" in capsys.readouterr().err.lower()
