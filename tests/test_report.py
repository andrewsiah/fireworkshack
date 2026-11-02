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


def test_safe_and_unsafe_refusals_are_not_merged(tmp_path):
    evaluation = {
        "arm": "A",
        "source": "measured",
        "scores": [
            {
                "item_id": "s",
                "suite": "xstest",
                "category": "safe",
                "refusal": False,
                "answer_complete": True,
            },
            {
                "item_id": "u",
                "suite": "xstest",
                "category": "unsafe",
                "refusal": True,
                "answer_complete": True,
            },
        ],
        "by_suite": {"xstest": {"total": 2}},
        "by_category": {"xstest/safe": {"total": 1}, "xstest/unsafe": {"total": 1}},
    }
    result = build_report(
        RunState(tmp_path / "events"), tmp_path / "report", evaluations=[evaluation]
    )
    rows = json.loads(result["json"].read_text())["metrics"]
    assert (
        next(
            r for r in rows if r["suite"] == "xstest/safe" and r["metric"] == "refusal"
        )["rate"]
        == 0
    )
    assert (
        next(
            r
            for r in rows
            if r["suite"] == "xstest/unsafe" and r["metric"] == "refusal"
        )["rate"]
        == 1
    )


def test_paired_capability_uses_same_ids(tmp_path):
    evaluations = [
        {
            "arm": arm,
            "source": "measured",
            "scores": [
                {"item_id": "mmlu:one", "suite": "mmlu", "capability_correct": correct}
            ],
        }
        for arm, correct in [("A", False), ("B", True)]
    ]
    result = build_report(
        RunState(tmp_path / "events"), tmp_path / "report", evaluations=evaluations
    )
    delta = json.loads(result["json"].read_text())["paired_capability"][0]
    assert (
        delta["pair"] == "B - A"
        and delta["matched_items"] == 1
        and delta["difference"] == 1
    )
