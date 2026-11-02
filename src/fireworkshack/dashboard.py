"""Loopback-only, read-only dashboard exposing reviewed aggregate fields.

No raw run files, credentials, prompts, generations, directory listings or
remote proxy endpoints are served. Missing/malformed sources yield a sanitized
stale snapshot instead of an exception containing private file contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .state import RunState

_TOKEN = re.compile(r"\b(?:fw_|abl_|hf_)[A-Za-z0-9_-]{8,}")
_JSON_LIMIT = 4 * 1024 * 1024
_EVENT_LIMIT = 64 * 1024 * 1024
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_LINKS = {
    "repo": "https://github.com/andrewsiah/fireworkshack",
    "plan": "https://github.com/andrewsiah/fireworkshack/blob/codex/experiment-plan/docs/superpowers/plans/2026-11-02-deepseek-comparison.md",
}


def _text(value, default="", limit=400):
    if not isinstance(value, str):
        return default
    return _TOKEN.sub("[redacted]", "".join(c for c in value if c >= " " or c == "\n"))[
        :limit
    ]


def _number(value, default=0, *, integral=False, maximum=1_000_000_000):
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return default
    if not math.isfinite(number) or number < 0 or number > maximum:
        return default
    if integral:
        return int(number) if number.is_integer() else default
    return number


def _invalid_constant(_value):
    raise ValueError("non-finite JSON number")


def _read_json(path, label, errors, *, required=False):
    try:
        if path.stat().st_size > _JSON_LIMIT:
            errors.append(label + "_too_large")
            return {}
        value = json.loads(
            path.read_text(encoding="utf8"), parse_constant=_invalid_constant
        )
        if not isinstance(value, dict):
            errors.append(label + "_invalid_schema")
            return {}
        return value
    except FileNotFoundError:
        if required:
            errors.append(label + "_missing")
    except (ValueError, UnicodeError):
        errors.append(label + "_invalid_json")
    except OSError:
        errors.append(label + "_unreadable")
    return {}


def _data(summary):
    counts = summary.get("evaluation_counts", {})
    counts = counts if isinstance(counts, dict) else {}
    pilot = counts.get("pilot", {})
    final = counts.get("final", {})
    pilot = pilot if isinstance(pilot, dict) else {}
    final = final if isinstance(final, dict) else {}
    return {
        "train": _number(summary.get("training_selected"), 64, integral=True),
        "dev": _number(summary.get("development_selected"), 16, integral=True),
        "train_tokens": _number(summary.get("train_tokens"), 8173, integral=True),
        "max_tokens": _number(summary.get("max_rendered_tokens"), 188, integral=True),
        "pilot": {
            suite: _number(pilot.get(suite), default, integral=True)
            for suite, default in (("mmlu", 57), ("harmbench", 16), ("xstest", 16))
        },
        "reserved_final": sum(
            _number(final.get(suite), default, integral=True)
            for suite, default in (("mmlu", 13985), ("harmbench", 304), ("xstest", 434))
        ),
        "reviewed": _number(summary.get("reviewed_examples"), 80, integral=True),
        "rejected": _number(
            summary.get("review_rejected", summary.get("rejected_examples")),
            42,
            integral=True,
        ),
    }


def _latest_run(root, errors):
    try:
        candidates = list((root / "runs").glob("*/controller.json"))
        return (
            max(candidates, key=lambda p: p.stat().st_mtime).parent
            if candidates
            else None
        )
    except OSError:
        errors.append("run_index_unreadable")
        return None


def _read_events(path, errors):
    try:
        if path.stat().st_size > _EVENT_LIMIT:
            errors.append("run_events_too_large")
            return []
        events = []
        with path.open(encoding="utf8") as file:
            for line in file:
                if not line.strip():
                    continue
                event = json.loads(line, parse_constant=_invalid_constant)
                if not isinstance(event, dict):
                    raise ValueError("invalid event schema")
                events.append(event)
        return events
    except (ValueError, UnicodeError):
        errors.append("run_events_invalid_json")
    except OSError:
        errors.append("run_events_unreadable")
    return []


def _verified_reference(root, binding, label, errors):
    """Validate a frozen private reference before parsing any of its bytes."""
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        errors.append(label + "_path_rejected")
        return None
    try:
        path = (root / binding["path"]).resolve()
        runs_root = (root / "runs").resolve()
        if not path.is_relative_to(runs_root) or path == runs_root:
            errors.append(label + "_path_rejected")
            return None
        expected = binding.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(label + "_integrity_mismatch")
            return None
        if path.stat().st_size > _EVENT_LIMIT:
            errors.append(label + "_too_large")
            return None
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            errors.append(label + "_integrity_mismatch")
            return None
        return path, content
    except (OSError, ValueError, RuntimeError):
        errors.append(label + "_unreadable")
        return None


def _generation_key(arm, result, fallback):
    """Deduplicate reused records and live events for the same locked item."""
    item = result.get("item_id")
    if isinstance(item, str) and isinstance(result.get("protocol_hash"), str):
        return arm, result["protocol_hash"], item
    return arm, "operation", str(result.get("operation_id", fallback))


def _reused_baselines(root, run, errors):
    generations, reused_arms = {}, set()
    if run is None:
        return generations, reused_arms
    index = _read_json(run / "baseline-sources.json", "baseline_sources", errors)
    for arm in ("A", "C"):
        for track in ("capability", "behavior"):
            name = f"{arm}-{track}"
            if name not in index:
                continue
            verified = _verified_reference(root, index[name], "baseline", errors)
            if verified is None:
                continue
            try:
                original = json.loads(verified[1], parse_constant=_invalid_constant)
                copied = _read_json(
                    run / (name + ".json"), "baseline_copy", errors, required=True
                )
                if (
                    not isinstance(original, dict)
                    or original.get("source") != "measured"
                    or original.get("arm") != arm
                    or copied.get("baseline_reused") is not True
                    or copied.get("baseline_source_sha256") != index[name]["sha256"]
                    or {
                        k: v
                        for k, v in copied.items()
                        if k not in {"baseline_reused", "baseline_source_sha256"}
                    }
                    != original
                ):
                    errors.append("baseline_copy_integrity_mismatch")
                    continue
                results = original.get("results")
                if not isinstance(results, list):
                    errors.append("baseline_invalid_schema")
                    continue
                for number, result in enumerate(results):
                    if (
                        not isinstance(result, dict)
                        or result.get("arm") != arm
                        or result.get("source") != "measured"
                        or result.get("status") not in ("completed", "error", "pending")
                    ):
                        errors.append("baseline_invalid_schema")
                        continue
                    generations[_generation_key(arm, result, name + str(number))] = (
                        result["status"]
                    )
                    reused_arms.add(arm)
            except (ValueError, TypeError, UnicodeError):
                errors.append("baseline_invalid_json")
    return generations, reused_arms


def _previous_costs(root, run, cost, current_paths, errors):
    if run is None:
        return
    index = _read_json(run / "previous-ledgers.json", "previous_ledgers", errors)
    seen = set()
    current_paths = {path.resolve() for path in current_paths}
    for path in current_paths:
        try:
            if path.stat().st_size <= _EVENT_LIMIT:
                seen.add(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            errors.append("run_accounting_unavailable")
    for provider in cost:
        if provider not in index:
            continue
        verified = _verified_reference(root, index[provider], "previous_ledger", errors)
        if verified is None:
            continue
        path, content = verified
        digest = hashlib.sha256(content).hexdigest()
        if path in current_paths or digest in seen:
            continue
        seen.add(digest)
        try:
            events = [
                json.loads(line, parse_constant=_invalid_constant)
                for line in content.splitlines()
                if line.strip()
            ]
            if any(not isinstance(row, dict) for row in events):
                raise ValueError("invalid ledger schema")
            # Replay verified bytes, avoiding a second unverified file read.
            _, spent, held = RunState._accounting(events)
            settled = _number(str(spent), None)
            reserved = _number(str(held), None)
            if settled is None or reserved is None:
                raise ValueError("invalid ledger amounts")
            cost[provider]["settled"] += settled
            cost[provider]["reserved"] += reserved
        except (ValueError, TypeError, KeyError, ArithmeticError):
            errors.append("previous_ledger_invalid_schema")


def _run_aggregates(root, run, errors, planned):
    cost = {
        "fireworks": {"cap": None, "reserved": 0, "settled": 0},
        "refuseless": {"cap": None, "reserved": 0, "settled": 0},
    }
    generations, reused_arms = _reused_baselines(root, run, errors)
    files = []
    if run:
        try:
            files = list(run.glob("*-events.jsonl"))
        except OSError:
            errors.append("run_events_unreadable")
            files = []
        for path in files:
            events = _read_events(path, errors)
            for index, event in enumerate(events):
                result = event.get("result")
                if event.get("type") != "generation" or not isinstance(result, dict):
                    continue
                source = result.get("source")
                if source is not None and not isinstance(source, str):
                    errors.append("run_events_invalid_schema")
                    continue
                if source not in {None, "measured"}:
                    continue
                arm = result.get("arm")
                status = result.get("status")
                if not isinstance(arm, str) or not isinstance(status, str):
                    errors.append("run_events_invalid_schema")
                    continue
                if arm not in {"A", "B", "C"} or status not in {
                    "completed",
                    "pending",
                    "error",
                }:
                    continue
                key = _generation_key(arm, result, event.get("operation_id", index))
                generations[key] = status
            provider = path.name.removesuffix("-events.jsonl")
            if provider not in cost or not events:
                continue
            try:
                account = RunState(path).summary()
                cost[provider]["reserved"] = _number(account.get("reserved_usd"))
                cost[provider]["settled"] = _number(account.get("spent_usd"))
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                ArithmeticError,
                OSError,
            ):
                errors.append("run_accounting_unavailable")
    _previous_costs(root, run, cost, files, errors)
    arms = []
    for arm, name, method, provider in (
        ("A", "Original DeepSeek", "Original weights", "Fireworks"),
        ("B", "Our Fireworks LoRA", "Benign-response SFT", "Fireworks"),
        ("C", "Refuseless comparison", "Hosted modified checkpoint", "Refuseless"),
    ):
        counts = Counter(status for key, status in generations.items() if key[0] == arm)
        status = "not_run"
        if counts["completed"] >= planned:
            status = "complete"
        elif counts["error"]:
            status = "partial" if counts["completed"] else "failed"
        elif counts["pending"]:
            status = "pending"
        elif counts["completed"]:
            status = "running"
        arms.append(
            {
                "id": arm,
                "name": name,
                "method": method,
                "provider": provider,
                "status": status,
                "completed": counts["completed"],
                "planned": planned,
                "failed": counts["error"],
                "pending": counts["pending"],
                "source": "reused baseline"
                if arm in reused_arms
                else "current run"
                if counts
                else "not run",
                "provenance": "Verified prior-run baseline reused; no replay."
                if arm in reused_arms
                else "",
            }
        )
    return arms, cost


def _metrics(run, errors):
    if not run:
        return []
    report = _read_json(run / "report/summary.json", "run_report", errors)
    if report.get("source") != "measured":
        return []
    if not isinstance(report.get("metrics"), list):
        errors.append("run_report_invalid_schema")
        return []
    metrics = []
    for row in report["metrics"][:300]:
        if not isinstance(row, dict) or not isinstance(row.get("arm"), str):
            errors.append("run_report_invalid_schema")
            continue
        if row.get("arm") not in {"A", "B", "C"}:
            continue
        metric = {
            key: _text(row.get(key), limit=80) for key in ("arm", "suite", "metric")
        }
        metric.update(
            {
                key: _number(row.get(key), integral=True)
                for key in ("successes", "scored", "total", "unknown")
            }
        )
        metric["rate"] = _number(row.get("rate"), None, maximum=1)
        interval = row.get("interval")
        metric["interval"] = (
            [_number(n, None, maximum=1) for n in interval]
            if isinstance(interval, list) and len(interval) == 2
            else None
        )
        metrics.append(metric)
    return metrics


def _execution_status(run, arms, errors):
    """Derive the hero from durable operations, never from model answer text."""
    if run is None:
        return None
    controller = _read_json(run / "controller.json", "run_controller", errors)
    status = controller.get("status")
    if status is not None and not isinstance(status, str):
        errors.append("run_controller_invalid_schema")
        return None
    if status == "incomplete":
        counts = "; ".join(
            f"{arm['id']}: {arm['completed']}/{arm['planned']} collected"
            for arm in arms
        )
        return {
            "status": "incomplete",
            "headline": "Pilot stopped with partial results",
            "summary": counts
            + ". Review recorded failures, reservations and exported partial results; unrun arms remain unmeasured.",
            "progress_status": "failed",
        }
    if status in {"completed", "completed_generations"}:
        return {
            "status": status,
            "headline": "Model generations complete",
            "summary": "Generation collection has finished. Independent behavioral judging and billing reconciliation remain separate from request completion.",
            "progress_status": "complete",
        }
    if status not in {"starting", "in_progress", "running"}:
        return None
    candidates = []
    completed_step = intended_step = 0
    observed = {
        "generation",
        "dispatch_intent",
        "training_step_intent",
        "training_step_complete",
        "training_session_intent",
        "training_session_allocated",
        "training_session_created",
        "checkpoint_saved",
        "controller_failure",
        "training_session_deleted",
        "failed_session_deleted",
        "session_finalized",
    }
    try:
        paths = list(run.glob("*-events.jsonl"))
        for path in paths:
            fallback_time = path.stat().st_mtime
            for index, event in enumerate(_read_events(path, errors)):
                kind = event.get("type")
                if not isinstance(kind, str) or kind not in observed:
                    continue
                arm = None
                if kind == "generation":
                    result = event.get("result", {})
                    if not isinstance(result, dict) or result.get("source") not in (
                        None,
                        "measured",
                    ):
                        continue
                    arm = result.get("arm")
                elif kind == "dispatch_intent":
                    arm = event.get("arm")
                if kind in {"generation", "dispatch_intent"} and arm not in (
                    "A",
                    "B",
                    "C",
                ):
                    continue
                if kind == "training_step_complete":
                    completed_step = max(
                        completed_step, _number(event.get("step"), integral=True)
                    )
                elif kind == "training_step_intent":
                    intended_step = max(
                        intended_step, _number(event.get("step"), integral=True)
                    )
                event_time = fallback_time + index / 1_000_000
                if isinstance(event.get("utc"), str):
                    try:
                        stamp = datetime.fromisoformat(
                            event["utc"].replace("Z", "+00:00")
                        )
                        event_time = (
                            stamp.replace(tzinfo=UTC).timestamp()
                            if stamp.tzinfo is None
                            else stamp.timestamp()
                        )
                    except (ValueError, OverflowError):
                        pass
                candidates.append((event_time, index, kind, arm))
    except OSError:
        errors.append("run_execution_unreadable")
    if not candidates:
        return {
            "status": "running",
            "headline": "Pilot starting",
            "summary": "The controller has started; waiting for the first recorded provider operation.",
            "progress_status": "running",
        }
    _, _, kind, arm_id = max(candidates)
    if kind in {"generation", "dispatch_intent"}:
        arm = next(row for row in arms if row["id"] == arm_id)
        return {
            "status": "running",
            "headline": f"Evaluating {arm_id}: {arm['completed']} of {arm['planned']} complete",
            "summary": "Collecting hosted model responses under the frozen protocol. Request completion is not a quality score; training and independent judging are tracked separately.",
            "progress_status": "running",
        }
    if kind in {"training_step_intent", "training_step_complete"}:
        detail = (
            f"Optimizer step {intended_step} has a recorded intent; waiting for its completion."
            if intended_step > completed_step
            else "The recorded optimizer update completed; collecting the next training checkpoint."
        )
        return {
            "status": "running",
            "headline": f"Training: {completed_step} of 8 steps complete",
            "summary": detail + " No post-training model comparison is claimed yet.",
            "progress_status": "running",
        }
    if kind == "checkpoint_saved":
        return {
            "status": "running",
            "headline": "Training checkpoint saved",
            "summary": "The training checkpoint is recorded. Post-training evaluation must finish before drawing a model comparison.",
            "progress_status": "running",
        }
    if kind == "controller_failure":
        return {
            "status": "incomplete",
            "headline": "Pilot interrupted; finalizing records",
            "summary": "A recorded controller failure stopped new work. Partial results are being exported and provider resources cleaned up.",
            "progress_status": "failed",
        }
    if kind in {
        "training_session_deleted",
        "failed_session_deleted",
        "session_finalized",
    }:
        return {
            "status": "running",
            "headline": "Finalizing pilot records",
            "summary": "Session finalization was recorded; waiting for the controller's final outcome and aggregate report.",
            "progress_status": "running",
        }
    return {
        "status": "running",
        "headline": "Preparing the Fireworks session",
        "summary": "The controller is preparing the supported training and sampling session. This is not evidence that a training update has completed.",
        "progress_status": "running",
    }


def build_status(root: Path) -> dict:
    """Read a fresh local aggregate snapshot; return only this explicit schema."""
    root = Path(root)
    errors = []
    summary = _read_json(
        root / "data/manifests/summary.json", "data_summary", errors, required=True
    )
    overlay = _read_json(
        root / ".local/dashboard-state.json", "dashboard_state", errors
    )
    data = _data(summary)
    run = _latest_run(root, errors)
    arms, cost = _run_aggregates(root, run, errors, sum(data["pilot"].values()))
    runtime_cost = overlay.get("cost", {})
    runtime_cost = runtime_cost if isinstance(runtime_cost, dict) else {}
    for provider, values in cost.items():
        supplied = runtime_cost.get(provider, {})
        values["cap"] = (
            _number(supplied.get("cap"), None) if isinstance(supplied, dict) else None
        )
    links = dict(_LINKS)
    runtime_links = overlay.get("links", {})
    task_link = runtime_links.get("linear") if isinstance(runtime_links, dict) else None
    if isinstance(task_link, str) and re.fullmatch(
        r"https://linear\.app/new-york-ai-labs/issue/AND-\d+(?:/[a-z0-9-]+)?", task_link
    ):
        links["linear"] = task_link
    metrics = _metrics(run, errors)
    verification = overlay.get("verification", {})
    verification = verification if isinstance(verification, dict) else {}
    progress = overlay.get("progress", [])
    progress = progress if isinstance(progress, list) else []
    updates = overlay.get("updates", [])
    updates = updates if isinstance(updates, list) else []
    risks = overlay.get("risks", [])
    risks = risks if isinstance(risks, list) else []
    execution = _execution_status(run, arms, errors)
    if execution:
        overlay = {
            **overlay,
            **{key: execution[key] for key in ("status", "headline", "summary")},
        }
        progress = [dict(row) for row in progress if isinstance(row, dict)]
        detail = execution["headline"] + ". " + execution["summary"]
        matching = [
            row
            for row in progress
            if str(row.get("label", "")).lower() in {"pilot execution", "execution"}
        ]
        if matching:
            matching[0].update(status=execution["progress_status"], detail=detail)
        else:
            progress.append(
                {
                    "label": "Pilot execution",
                    "status": execution["progress_status"],
                    "detail": detail,
                }
            )
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "status": _text(overlay.get("status"), "prepared", limit=60),
        "headline": _text(
            overlay.get("headline"), "Pilot prepared · model run not started", limit=180
        ),
        "summary": _text(
            overlay.get("summary"),
            "Data preparation and offline tests are separate from measured model performance.",
            limit=700,
        ),
        "progress": [
            {
                "label": _text(row.get("label"), limit=120),
                "status": _text(row.get("status"), limit=60),
                "detail": _text(row.get("detail"), limit=300),
            }
            for row in progress[:20]
            if isinstance(row, dict)
        ],
        "data": data,
        "verification": {
            "tests": _number(verification.get("tests"), integral=True),
            "passed": verification["passed"]
            if isinstance(verification.get("passed"), bool)
            else _number(verification.get("passed"), integral=True),
        },
        "arms": arms,
        "metrics": metrics,
        "updates": [
            {
                "time": _text(row.get("time"), limit=60),
                "text": _text(row.get("text"), limit=500),
            }
            for row in updates[-30:]
            if isinstance(row, dict)
        ],
        "cost": cost,
        "risks": [
            _text(risk, limit=500) for risk in risks[:15] if isinstance(risk, str)
        ],
        "links": links,
        "stale": bool(errors),
        "errors": sorted(set(errors)),
    }


def create_server(root: Path, *, port: int = 8765) -> ThreadingHTTPServer:
    """Construct a loopback server; the caller owns startup and shutdown."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("invalid dashboard port")
    root = Path(root).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "FireworksHack"
        sys_version = ""

        def log_message(self, _format, *_args):
            # Request URLs can contain accidentally pasted secrets. No access log.
            pass

        def do_GET(self):
            self._serve()

        def do_HEAD(self):
            self._serve(head=True)

        def _serve(self, head=False):
            try:
                parsed = urlsplit(self.path)
                route = unquote(parsed.path)
            except ValueError:
                self._reply(404, b"Not found", "text/plain", head=head)
                return
            if parsed.scheme or parsed.netloc:
                self._reply(404, b"Not found", "text/plain", head=head)
                return
            if route == "/api/status":
                body = json.dumps(build_status(root), allow_nan=False).encode()
                self._reply(200, body, "application/json; charset=utf-8", head=head)
            elif route in _ASSETS:
                name, content_type = _ASSETS[route]
                web = root / "web"
                asset = web / name
                try:
                    if web.is_symlink():
                        raise FileNotFoundError
                    fd = os.open(asset, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(fd, "rb") as file:
                        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                            raise FileNotFoundError
                        body = file.read(_JSON_LIMIT + 1)
                    if len(body) > _JSON_LIMIT:
                        raise FileNotFoundError
                    self._reply(200, body, content_type, head=head)
                except OSError:
                    self._reply(404, b"Not found", "text/plain", head=head)
            else:
                self._reply(404, b"Not found", "text/plain", head=head)

        def _reply(self, code, body, content_type, *, head=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _method_not_allowed(self):
            self._reply(405, b"Method not allowed", "text/plain")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _method_not_allowed

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local read-only FireworksHack dashboard"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(Path(__file__).resolve().parents[2], port=args.port)
    print(
        f"FireworksHack dashboard: http://127.0.0.1:{server.server_address[1]}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
