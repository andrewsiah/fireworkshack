import json

import pytest

from fireworkshack.report import build_report, wilson
from fireworkshack.state import RunState


def test_no_run_is_not_zero_accuracy_and_chart_is_real(tmp_path):
    state = RunState(tmp_path / "events.jsonl")
    result = build_report(
        state,
        tmp_path / "report",
        readiness={"HF dataset access": True, "Training run": False},
    )
    body = result["markdown"].read_text()
    assert "not run" in body
    assert "0.0%" not in body
    assert result["chart"].stat().st_size > 1000
    assert json.loads(result["json"].read_text())["source"] == "readiness"


def test_synthetic_and_author_scores_cannot_be_measured(tmp_path):
    state = RunState(tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="source"):
        build_report(
            state,
            tmp_path / "report",
            evaluations=[{"arm": "C", "source": "author_reported", "scores": []}],
        )


def test_wilson_edges():
    assert wilson(0, 0) is None
    lo, hi = wilson(1, 1)
    assert 0 < lo < 1 and hi == pytest.approx(1)


def test_partial_coverage_cannot_look_like_full_accuracy(tmp_path):
    result = build_report(
        RunState(tmp_path / "events"),
        tmp_path / "report",
        evaluations=[
            {
                "arm": "A",
                "source": "measured",
                "total": 57,
                "missing": 56,
                "completed": 1,
                "by_suite": {
                    "mmlu": {"total": 57, "missing": 56, "completed": 1, "errors": 0}
                },
                "scores": [
                    {
                        "item_id": "mmlu:one",
                        "suite": "mmlu",
                        "capability_correct": True,
                        "answer_complete": True,
                        "truncated": False,
                    }
                ],
            }
        ],
    )
    summary = json.loads(result["json"].read_text())
    metric = next(r for r in summary["metrics"] if r["metric"] == "capability")
    assert metric["total"] == 57
    assert metric["rate"] == pytest.approx(1 / 57)
    assert summary["coverage"][0]["missing"] == 56
