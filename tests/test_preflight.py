from pathlib import Path

import pytest

from fireworkshack.config import load_config, preflight, validate_config


def test_offline_preflight_never_claims_remote_access():
    config = load_config(Path("configs/pilot.yaml"))
    report = preflight(
        config, {"FIREWORKS_API_KEY": "SECRET", "REFUSELESS_API_KEY": "SECRET"}
    )
    assert report["network_calls"] == 0
    assert report["checks"]["fireworks_capacity"] == "unverified"
    assert "SECRET" not in str(report)
    assert report["paid_ready"] is False


def test_missing_key_names_only():
    report = preflight(load_config(Path("configs/pilot.yaml")), {})
    assert report["missing_credentials"] == ["FIREWORKS_API_KEY", "REFUSELESS_API_KEY"]


@pytest.mark.parametrize(
    "changes",
    [
        {"budget_usd": "-1"},
        {"request_ceiling": 0},
        {"max_retries": -1},
        {"request_ceiling": True},
        {"request_ceiling": 1.5},
        {"max_retries": True},
        {"paid_execution": True, "budget_usd": None},
        {"paid_execution": True, "budget_usd": "5", "deadline_utc": None},
    ],
)
def test_invalid_limits_rejected(changes):
    config = load_config(Path("configs/pilot.yaml"))
    config.update(changes)
    with pytest.raises(ValueError):
        validate_config(config)
