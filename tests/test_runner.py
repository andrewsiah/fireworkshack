import hashlib
import json

import pytest

from fireworkshack.runner import (
    exclusive_controller,
    load_private_manifest,
    pool_summary,
)
from fireworkshack.state import RunState


def test_modified_private_data_is_rejected_before_dispatch(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"ready": True}))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_private_manifest(path, expected)["ready"] is True
    path.write_text(json.dumps({"ready": False}))
    with pytest.raises(ValueError, match="hash"):
        load_private_manifest(path, expected)


def test_provider_pool_summary_keeps_unsettled_funds(tmp_path):
    state = RunState(tmp_path / "events")
    summary = pool_summary({"fireworks": state})
    assert summary["spent_usd"] == "0"
    assert summary["reserved_usd"] == "0"
    assert "fireworks" in summary["providers"]


def test_second_controller_cannot_acquire_live_lease(tmp_path):
    with exclusive_controller(tmp_path / "controller.lock"):
        with pytest.raises(ValueError, match="controller"):
            with exclusive_controller(tmp_path / "controller.lock"):
                pytest.fail("duplicate controller acquired ownership")
