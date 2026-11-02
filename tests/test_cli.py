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


def test_report_reads_saved_measurements(tmp_path, capsys):
    result = {
        "arm": "C",
        "source": "measured",
        "total": 1,
        "completed": 1,
        "missing": 0,
        "scores": [
            {
                "item_id": "mmlu:fixture",
                "suite": "mmlu",
                "capability_correct": True,
                "answer_complete": True,
                "truncated": False,
                "status": "completed",
            }
        ],
        "by_suite": {"mmlu": {"total": 1, "completed": 1, "missing": 0, "errors": 0}},
    }
    (tmp_path / "C-capability.json").write_text(json.dumps(result))
    assert main(["report", "--run-dir", str(tmp_path)]) == 0
    artifact = json.loads(capsys.readouterr().out)
    summary = json.loads(open(artifact["json"]).read())
    assert summary["arms"]["C"] == "measured"
