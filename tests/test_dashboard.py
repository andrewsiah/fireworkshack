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
