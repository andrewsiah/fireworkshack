import hashlib
import json
import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from fireworkshack.dashboard import build_status, create_server


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@contextmanager
def running(root):
    server = create_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_api_whitelists_data_and_status_fields(tmp_path):
    put_json(
        tmp_path / "data/manifests/summary.json",
        {
            "training_selected": 64,
            "development_selected": 16,
            "train_tokens": 8173,
            "max_rendered_tokens": 188,
            "reviewed_examples": 80,
            "evaluation_counts": {
                "pilot": {"mmlu": 57, "harmbench": 16, "xstest": 16},
                "final": {"mmlu": 13985, "harmbench": 304, "xstest": 434},
            },
            "FIREWORKS_API_KEY": "fw_fakeSecretValue123",
            "prompt": "PRIVATE_PROMPT",
        },
    )
    put_json(
        tmp_path / ".local/dashboard-state.json",
        {
            "status": "running",
            "headline": "Checking fw_fakeSecretValue123",
            "summary": "Private key abl_fakeSecretValue123 must be hidden",
            "credential": "PRIVATE_CREDENTIAL",
            "response": "PRIVATE_RESPONSE",
            "progress": [
                {
                    "label": "Prepared",
                    "status": "complete",
                    "detail": "Ready",
                    "prompt": "PRIVATE_PROMPT",
                }
            ],
            "verification": {
                "tests": 120,
                "passed": 120,
                "token": "PRIVATE_CREDENTIAL",
            },
            "updates": [
                {
                    "time": "2026-11-02T20:00:00Z",
                    "text": "Prepared",
                    "response": "PRIVATE_RESPONSE",
                }
            ],
        },
    )
    snapshot = build_status(tmp_path)
    text = json.dumps(snapshot)
    assert snapshot["data"]["train"] == 64
    assert snapshot["data"]["reserved_final"] == 14723
    assert snapshot["verification"] == {"tests": 120, "passed": 120}
    assert snapshot["status"] == "running"
    assert all(
        secret not in text
        for secret in [
            "fw_fakeSecretValue123",
            "abl_fakeSecretValue123",
            "PRIVATE_PROMPT",
            "PRIVATE_RESPONSE",
            "PRIVATE_CREDENTIAL",
        ]
    )


@pytest.mark.parametrize(
    "path",
    [
        "/.env",
        "/../.env",
        "/%2e%2e/.env",
        "/web/",
        "/data/manifests/summary.json",
        "/api/status/../.env",
        "/%252e%252e/.env",
    ],
)
def test_unknown_files_and_traversal_are_not_served(tmp_path, path):
    (tmp_path / ".env").write_text("private secret")
    with running(tmp_path) as base, pytest.raises(HTTPError) as error:
        urlopen(base + path)
    assert error.value.code == 404


def test_static_whitelist_and_api_are_served_loopback_only(tmp_path):
    (tmp_path / "web").mkdir()
    for name, contents in [
        ("index.html", "<h1>Experiment</h1>"),
        ("styles.css", "body{}"),
        ("app.js", "console.log('ready')"),
    ]:
        (tmp_path / "web" / name).write_text(contents)
    server = create_server(tmp_path, port=0)
    assert server.server_address[0] == "127.0.0.1"
    server.server_close()
    with running(tmp_path) as base:
        assert urlopen(base + "/").read() == b"<h1>Experiment</h1>"
        response = urlopen(base + "/api/status")
        assert response.headers["Cache-Control"] == "no-store"
        assert "headline" in json.load(response)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/status", data=b"{}", method="POST"))
        assert error.value.code == 405


def test_static_symlink_cannot_expose_private_file(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / ".env").write_text("private secret")
    (tmp_path / "web/app.js").symlink_to(tmp_path / ".env")
    with running(tmp_path) as base, pytest.raises(HTTPError) as error:
        urlopen(base + "/app.js")
    assert error.value.code == 404


def test_malformed_json_is_stale_and_does_not_return_raw_error(tmp_path):
    path = tmp_path / ".local/dashboard-state.json"
    path.parent.mkdir()
    path.write_text('{"key":"PRIVATE_CREDENTIAL", broken')
    snapshot = build_status(tmp_path)
    assert snapshot["stale"] is True
    assert "dashboard_state_invalid_json" in snapshot["errors"]
    assert "PRIVATE_CREDENTIAL" not in json.dumps(snapshot)


def test_latest_run_aggregates_counts_costs_and_safe_metrics(tmp_path):
    run = tmp_path / "runs/pilot-1"
    put_json(
        run / "controller.json", {"status": "running", "secret": "PRIVATE_CREDENTIAL"}
    )
    events = [
        {"type": "reserve", "operation_id": "1", "maximum_usd": "0.07"},
        {
            "type": "generation",
            "result": {
                "operation_id": "1",
                "arm": "A",
                "status": "completed",
                "final_text": "PRIVATE_RESPONSE",
            },
        },
        {"type": "reserve", "operation_id": "2", "maximum_usd": "0.03"},
        {"type": "settle", "operation_id": "2", "actual_usd": "0.01"},
        {
            "type": "generation",
            "result": {
                "operation_id": "2",
                "arm": "B",
                "status": "error",
                "reasoning": "PRIVATE_RESPONSE",
            },
        },
    ]
    (run / "fireworks-events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    put_json(
        run / "report/summary.json",
        {
            "source": "measured",
            "metrics": [
                {
                    "arm": "A",
                    "suite": "mmlu",
                    "metric": "capability",
                    "successes": 1,
                    "scored": 1,
                    "total": 57,
                    "unknown": 56,
                    "rate": 1.0,
                    "interval": [0.2, 1.0],
                    "prompt": "PRIVATE_PROMPT",
                }
            ],
        },
    )
    snapshot = build_status(tmp_path)
    assert snapshot["arms"][0]["completed"] == 1
    assert snapshot["arms"][1]["failed"] == 1
    assert snapshot["cost"]["fireworks"]["reserved"] == pytest.approx(0.07)
    assert snapshot["cost"]["fireworks"]["settled"] == pytest.approx(0.01)
    assert snapshot["metrics"][0]["total"] == 57
    assert "PRIVATE_" not in json.dumps(snapshot)


def test_synthetic_report_is_never_measured_dashboard_metrics(tmp_path):
    run = tmp_path / "runs/pilot-1"
    put_json(run / "controller.json", {})
    put_json(
        run / "report/summary.json", {"source": "synthetic", "metrics": [{"rate": 1}]}
    )
    assert build_status(tmp_path)["metrics"] == []


def test_malformed_source_schema_returns_sanitized_stale_snapshot(tmp_path):
    run = tmp_path / "runs/pilot-1"
    put_json(run / "controller.json", {})
    (run / "fireworks-events.jsonl").write_text(
        json.dumps(
            {
                "type": "generation",
                "result": {
                    "arm": ["PRIVATE_CREDENTIAL"],
                    "status": "completed",
                    "source": {"secret": "PRIVATE_CREDENTIAL"},
                },
            }
        )
    )
    put_json(
        run / "report/summary.json",
        {"source": "measured", "metrics": [{"arm": ["PRIVATE_CREDENTIAL"]}]},
    )
    snapshot = build_status(tmp_path)
    assert snapshot["stale"] is True
    assert "run_events_invalid_schema" in snapshot["errors"]
    assert "run_report_invalid_schema" in snapshot["errors"]
    assert "PRIVATE_CREDENTIAL" not in json.dumps(snapshot)


def test_boolean_verification_result_preserves_frontend_contract(tmp_path):
    put_json(
        tmp_path / ".local/dashboard-state.json",
        {"verification": {"tests": 101, "passed": True}},
    )
    assert build_status(tmp_path)["verification"] == {"tests": 101, "passed": True}


def test_live_events_override_old_preparation_headline_without_implying_training(
    tmp_path,
):
    put_json(
        tmp_path / ".local/dashboard-state.json",
        {
            "headline": "Baseline starts next",
            "status": "prepared",
            "progress": [
                {"label": "Training data", "status": "complete", "detail": "Reviewed"},
                {
                    "label": "Pilot execution",
                    "status": "pending",
                    "detail": "Not started",
                },
            ],
        },
    )
    run = tmp_path / "runs/pilot-1"
    put_json(run / "controller.json", {"status": "starting"})
    events = [
        {
            "utc": f"2026-11-02T21:00:{n:02d}+00:00",
            "type": "generation",
            "result": {
                "operation_id": str(n),
                "arm": "C",
                "status": "completed",
                "source": "measured",
            },
        }
        for n in range(40)
    ]
    (run / "refuseless-events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    snapshot = build_status(tmp_path)
    assert snapshot["status"] == "running"
    assert snapshot["headline"] == "Evaluating C: 40 of 89 complete"
    assert snapshot["arms"][1]["completed"] == 0
    assert "training completed" not in snapshot["summary"].lower()
    assert snapshot["progress"][0]["label"] == "Training data"
    assert snapshot["progress"][1]["status"] == "running"
    assert "40 of 89" in snapshot["progress"][1]["detail"]


def test_training_headline_uses_actual_completed_step_events(tmp_path):
    run = tmp_path / "runs/pilot-1"
    put_json(run / "controller.json", {"status": "in_progress"})
    (run / "fireworks-events.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "utc": "2026-11-02T21:00:00+00:00",
                    "type": "training_step_complete",
                    "step": 3,
                },
                {
                    "utc": "2026-11-02T21:00:01+00:00",
                    "type": "training_step_intent",
                    "step": 4,
                },
            ]
        )
    )
    snapshot = build_status(tmp_path)
    assert snapshot["status"] == "running"
    assert snapshot["headline"] == "Training: 3 of 8 steps complete"
    assert "step 4" in snapshot["summary"]


@pytest.mark.parametrize(
    "status,expected",
    [
        ("incomplete", "Pilot stopped with partial results"),
        ("completed_generations", "Model generations complete"),
    ],
)
def test_final_controller_status_overrides_static_or_prior_live_headline(
    tmp_path, status, expected
):
    put_json(
        tmp_path / ".local/dashboard-state.json",
        {"headline": "Baseline starts next", "status": "prepared"},
    )
    put_json(
        tmp_path / "runs/pilot-1/controller.json",
        {"status": status, "exception_type": "PRIVATE_CREDENTIAL"},
    )
    snapshot = build_status(tmp_path)
    assert snapshot["status"] == status
    assert snapshot["headline"] == expected
    assert "PRIVATE_CREDENTIAL" not in json.dumps(snapshot)


def test_invalid_controller_status_is_a_stale_source_not_server_error(tmp_path):
    put_json(
        tmp_path / "runs/pilot-1/controller.json",
        {"status": {"secret": "PRIVATE_CREDENTIAL"}},
    )
    snapshot = build_status(tmp_path)
    assert snapshot["stale"] is True
    assert "run_controller_invalid_schema" in snapshot["errors"]
    assert "PRIVATE_CREDENTIAL" not in json.dumps(snapshot)


def test_runtime_caps_and_private_task_link_have_no_public_defaults(tmp_path):
    snapshot = build_status(tmp_path)
    assert snapshot["cost"]["fireworks"]["cap"] is None
    assert snapshot["cost"]["refuseless"]["cap"] is None
    assert "linear" not in snapshot["links"]
    put_json(
        tmp_path / ".local/dashboard-state.json",
        {
            "cost": {
                "fireworks": {"cap": 13.25, "settled": 999},
                "refuseless": {"cap": 4.5},
            },
            "links": {
                "linear": "https://linear.app/new-york-ai-labs/issue/AND-123",
                "secret": "PRIVATE_CREDENTIAL",
            },
        },
    )
    snapshot = build_status(tmp_path)
    assert snapshot["cost"]["fireworks"]["cap"] == 13.25
    assert snapshot["cost"]["refuseless"]["cap"] == 4.5
    assert snapshot["cost"]["fireworks"]["settled"] == 0
    assert snapshot["links"]["linear"].endswith("AND-123")
    assert "PRIVATE_CREDENTIAL" not in json.dumps(snapshot)


@pytest.mark.parametrize("cap", [-1, True, "NaN", "Infinity", {"private": "secret"}])
def test_invalid_overlay_cap_stays_unknown(tmp_path, cap):
    put_json(
        tmp_path / ".local/dashboard-state.json", {"cost": {"fireworks": {"cap": cap}}}
    )
    assert build_status(tmp_path)["cost"]["fireworks"]["cap"] is None


def write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in events))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def recovered_baselines(root, run):
    bindings = {}
    for arm in ["A", "C"]:
        for track, count in [("capability", 57), ("behavior", 32)]:
            name = f"{arm}-{track}"
            original = {
                "arm": arm,
                "source": "measured",
                "results": [
                    {
                        "item_id": f"{track}-{n}",
                        "operation_id": f"{arm}-{track}-old-{n}",
                        "protocol_hash": track,
                        "status": "completed",
                        "arm": arm,
                        "source": "measured",
                    }
                    for n in range(count)
                ],
            }
            path = root / "runs/previous" / f"{name}.json"
            put_json(path, original)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            bindings[name] = {"path": str(path), "sha256": sha}
            put_json(
                run / f"{name}.json",
                {**original, "baseline_reused": True, "baseline_source_sha256": sha},
            )
    put_json(run / "baseline-sources.json", bindings)
    return bindings


def test_recovery_reuses_verified_baseline_counts_and_rolls_up_costs_once(tmp_path):
    run = tmp_path / "runs/recovery"
    recovered_baselines(tmp_path, run)
    prior_fw = write_events(
        tmp_path / "runs/previous/fireworks-events.jsonl",
        [
            {"type": "reserve", "operation_id": "old-known", "maximum_usd": "2"},
            {"type": "settle", "operation_id": "old-known", "actual_usd": "0.5"},
            {"type": "reserve", "operation_id": "old-unknown", "maximum_usd": "3"},
        ],
    )
    prior_rf = write_events(
        tmp_path / "runs/previous/refuseless-events.jsonl",
        [
            {"type": "reserve", "operation_id": "old-rf", "maximum_usd": "0.1"},
            {"type": "settle", "operation_id": "old-rf", "actual_usd": "0.02"},
        ],
    )
    put_json(
        run / "previous-ledgers.json", {"fireworks": prior_fw, "refuseless": prior_rf}
    )
    write_events(
        run / "fireworks-events.jsonl",
        [
            {"type": "reserve", "operation_id": "new-step", "maximum_usd": "1"},
            {
                "type": "generation",
                "result": {
                    "arm": "A",
                    "item_id": "capability-0",
                    "operation_id": "new-duplicate",
                    "protocol_hash": "capability",
                    "status": "completed",
                    "source": "measured",
                },
            },
        ],
    )
    put_json(run / "controller.json", {"run_id": run.name, "status": "starting"})
    snapshot = build_status(tmp_path)
    assert [row["completed"] for row in snapshot["arms"]] == [89, 0, 89]
    assert snapshot["arms"][0]["source"] == "reused baseline"
    assert snapshot["arms"][2]["source"] == "reused baseline"
    assert snapshot["cost"]["fireworks"]["reserved"] == pytest.approx(4)
    assert snapshot["cost"]["fireworks"]["settled"] == pytest.approx(0.5)
    assert snapshot["cost"]["refuseless"]["settled"] == pytest.approx(0.02)
    text = json.dumps(snapshot)
    assert str(tmp_path) not in text
    assert "new-duplicate" not in text


def test_previous_ledger_hash_and_containment_are_required(tmp_path):
    run = tmp_path / "runs/recovery"
    put_json(run / "controller.json", {"status": "starting"})
    outside = write_events(
        tmp_path / "private-events.jsonl",
        [{"type": "reserve", "operation_id": "PRIVATE_OP", "maximum_usd": "100"}],
    )
    inside = write_events(
        tmp_path / "runs/previous/fireworks-events.jsonl",
        [{"type": "reserve", "operation_id": "PRIVATE_OP", "maximum_usd": "100"}],
    )
    inside["sha256"] = "0" * 64
    put_json(
        run / "previous-ledgers.json", {"fireworks": inside, "refuseless": outside}
    )
    snapshot = build_status(tmp_path)
    assert snapshot["stale"] is True
    assert "previous_ledger_integrity_mismatch" in snapshot["errors"]
    assert "previous_ledger_path_rejected" in snapshot["errors"]
    assert snapshot["cost"]["fireworks"]["reserved"] == 0
    assert snapshot["cost"]["refuseless"]["reserved"] == 0
    assert "PRIVATE_OP" not in json.dumps(snapshot)


def test_previous_ledger_same_hash_is_not_counted_twice(tmp_path):
    run = tmp_path / "runs/recovery"
    put_json(run / "controller.json", {"status": "starting"})
    prior = write_events(
        tmp_path / "runs/previous/fireworks-events.jsonl",
        [{"type": "reserve", "operation_id": "old", "maximum_usd": "1"}],
    )
    put_json(run / "previous-ledgers.json", {"fireworks": prior, "refuseless": prior})
    snapshot = build_status(tmp_path)
    assert sum(value["reserved"] for value in snapshot["cost"].values()) == 1


def test_reused_baseline_mutation_cannot_spoof_verified_counts(tmp_path):
    run = tmp_path / "runs/recovery"
    bindings = recovered_baselines(tmp_path, run)
    put_json(run / "controller.json", {"run_id": run.name, "status": "starting"})
    source = bindings["A-capability"]["path"]
    with open(source, "a") as file:
        file.write(" ")
    snapshot = build_status(tmp_path)
    assert snapshot["stale"] is True
    assert "baseline_integrity_mismatch" in snapshot["errors"]
    assert snapshot["arms"][0]["completed"] == 32
