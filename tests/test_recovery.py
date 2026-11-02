import json

import pytest

from fireworkshack.runner import verify_recovery_baselines


def test_recovery_rejects_unbound_or_changed_baseline(tmp_path):
    with pytest.raises(ValueError, match="baseline"):
        verify_recovery_baselines({}, {"pilot": []})


def test_recovery_requires_unchanged_items_and_protocol(tmp_path):
    import hashlib

    from fireworkshack.providers import protocol_hash

    config = {
        "baseline_files": {},
        "tracks": {"capability": {"fixed": True}, "behavior": {"fixed": True}},
        "arms": {"A": {"model": "base"}, "C": {"model": "modified"}},
    }
    manifest = {"pilot": [{"item_id": "mmlu:fixture", "suite": "mmlu"}]}
    for arm in ("A", "C"):
        for track in ("capability", "behavior"):
            items = manifest["pilot"] if track == "capability" else []
            result = {
                "arm": arm,
                "source": "measured",
                "items": items,
                "protocol_hash": protocol_hash(config["tracks"][track]),
                "results": [{"model": config["arms"][arm]["model"]} for _ in items],
            }
            path = tmp_path / f"{arm}-{track}.json"
            path.write_text(json.dumps(result))
            config["baseline_files"][f"{arm}-{track}"] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    assert len(verify_recovery_baselines(config, manifest)) == 4
    config["tracks"]["capability"]["fixed"] = False
    with pytest.raises(ValueError, match="protocol"):
        verify_recovery_baselines(config, manifest)
