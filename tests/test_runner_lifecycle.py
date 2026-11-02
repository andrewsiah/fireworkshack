"""Synthetic lifecycle checks; every provider/session boundary is mocked."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from fireworkshack import runner


@pytest.fixture
def pilot(monkeypatch, tmp_path):
    service = SimpleNamespace(training_session_id="synthetic-session", deleted=False)
    captured_arms = []

    def delete_session():
        service.deleted = True

    service.delete_training_session = delete_session
    checkpoint = SimpleNamespace(
        sampler_path="synthetic-checkpoint",
        to_dict=lambda: {"sampler_path": "synthetic-checkpoint"},
    )
    callback = SimpleNamespace(close=lambda: None)
    config = {
        "paid_execution": True,
        "data_manifest": "synthetic-train",
        "data_sha256": "synthetic-hash",
        "eval_manifest": "synthetic-eval",
        "eval_sha256": "synthetic-hash",
        "checkpoint_retention": "delete_after_report",
        "provider_budgets": {"fireworks": "0.02", "refuseless": "0.02"},
        "budget_usd": "0.04",
        "deadline_utc": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        "request_ceiling": 300,
        "max_retries": 0,
        "run_id": "synthetic-lifecycle",
        "arms": {
            "C": {"model": "synthetic-c"},
            "A": {
                "model": "synthetic-a",
                "maximum_cost_multiplier": 1,
                "usage_accounts_for_all_attempts": True,
            },
        },
        "tracks": {"capability": {}, "behavior": {}},
        "training": {"base_model": "synthetic-base"},
    }
    approval = {"approval_quote": "synthetic test only", "config_sha256": "synthetic"}

    def load_manifest(path, _expected_hash):
        if path == "synthetic-eval":
            return {"pilot": [{"suite": "mmlu"}] * 89}
        return {}

    def evaluate(arm, *_args, **_kwargs):
        captured_arms.append(dict(arm))
        return {
            "arm": arm["name"],
            "source": "measured",
            "scores": [],
            "errors": 0,
            "pending": 0,
            "missing": 0,
        }

    monkeypatch.setattr(runner, "load_private_manifest", load_manifest)
    monkeypatch.setattr(runner, "load_runtime_credentials", lambda: None)
    monkeypatch.setattr(runner.NativeRenderer, "load", lambda **_kwargs: object())
    monkeypatch.setattr(
        runner,
        "train_sft",
        lambda *_args, dry_run=True, **_kwargs: {} if dry_run else checkpoint,
    )
    monkeypatch.setattr(
        runner, "open_session", lambda *_args: {"service": service, "client": None}
    )
    monkeypatch.setattr(runner, "make_base_sampler", lambda *_args: callback)
    monkeypatch.setattr(runner, "make_checkpoint_sampler", lambda *_args: callback)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(
        runner, "finalize_session", lambda *_args, **_kwargs: delete_session()
    )
    monkeypatch.setattr(
        runner,
        "build_report",
        lambda *_args, **_kwargs: {"markdown": tmp_path / "report.md"},
    )
    return config, approval, service, captured_arms, tmp_path / "run"


def test_report_export_failure_still_deletes_known_session(pilot, monkeypatch):
    config, approval, service, _arms, directory = pilot

    def fail_report(*_args, **_kwargs):
        raise OSError("synthetic report storage failure")

    monkeypatch.setattr(runner, "build_report", fail_report)
    try:
        runner.run_pilot(config, directory, approval)
    except OSError:
        # Either surfacing the export error or returning an incomplete result is
        # acceptable; skipping known-resource cleanup is not.
        pass
    assert service.deleted, "report failure must not skip session cleanup"


def test_sdk_arms_force_complete_attempt_reservation(pilot):
    config, approval, _service, captured, directory = pilot
    runner.run_pilot(config, directory, approval)
    sdk_arms = [arm for arm in captured if arm["name"] in {"A", "B"}]
    assert {arm["name"] for arm in sdk_arms} == {"A", "B"}
    assert all(arm["maximum_cost_multiplier"] == 7 for arm in sdk_arms)
    assert all(arm["usage_accounts_for_all_attempts"] is False for arm in sdk_arms)


def test_recovery_does_not_replay_paid_baselines(pilot,monkeypatch):
    config,approval,_service,captured,directory=pilot
    config['baseline_files']={'explicit_recovery':True}
    records=[(arm,track,{'arm':arm,'source':'measured','scores':[],'baseline_source_sha256':'test'})
             for arm in ('A','C') for track in ('capability','behavior')]
    monkeypatch.setattr(runner,'verify_recovery_baselines',lambda *_args:records)
    runner.run_pilot(config,directory,approval)
    assert captured and {a['name'] for a in captured}=={'B'}
    assert (directory/'A-capability.json').exists()
    assert (directory/'C-behavior.json').exists()
