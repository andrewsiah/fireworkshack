"""One approved, bounded pilot. Raw run records always stay private."""

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from .evaluate import evaluate
from .providers import protocol_hash
from .report import build_report
from .state import RunState
from .train import (
    NativeRenderer,
    finalize_session,
    make_base_sampler,
    make_checkpoint_sampler,
    open_session,
    train_sft,
)


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
    ) as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_private_manifest(path, expected_sha256):
    raw = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("private manifest hash differs from approved plan")
    return json.loads(raw)


def pool_summary(states):
    summaries = {name: state.summary() for name, state in states.items()}
    return {
        "providers": summaries,
        "spent_usd": str(
            sum((Decimal(x["spent_usd"]) for x in summaries.values()), Decimal(0))
        ),
        "reserved_usd": str(
            sum((Decimal(x["reserved_usd"]) for x in summaries.values()), Decimal(0))
        ),
    }


class CombinedLedger:
    def __init__(self, states):
        self.states = states

    def summary(self):
        return pool_summary(self.states)


class LedgerGroup:
    def __init__(self, *ledgers):
        self.ledgers = ledgers

    def summary(self):
        values = [s.summary() for s in self.ledgers]
        return {
            "spent_usd": str(
                sum((Decimal(s["spent_usd"]) for s in values), Decimal(0))
            ),
            "reserved_usd": str(
                sum((Decimal(s["reserved_usd"]) for s in values), Decimal(0))
            ),
            "operation_count": sum(s["operation_count"] for s in values),
            "pending_operations": [
                op for s in values for op in s.get("pending_operations", [])
            ],
            "budget_breaches": [
                e for s in values for e in s.get("budget_breaches", [])
            ],
        }


def verify_recovery_baselines(config, manifest):
    files = config.get("baseline_files", {})
    if set(files) != {
        f"{arm}-{track}" for arm in ("A", "C") for track in ("capability", "behavior")
    }:
        raise ValueError("all four hash-bound baseline files required")
    records = []
    for arm in ("A", "C"):
        for track in ("capability", "behavior"):
            ref = files[f"{arm}-{track}"]
            record = load_private_manifest(ref["path"], ref["sha256"])
            if record.get("source") != "measured" or record.get("arm") != arm:
                raise ValueError("baseline source or arm differs")
            if record.get("protocol_hash") != protocol_hash(config["tracks"][track]):
                raise ValueError("baseline protocol differs")
            items = [
                i
                for i in manifest["pilot"]
                if (i["suite"] == "mmlu") == (track == "capability")
            ]
            if record.get("items") != items or len(record.get("results", [])) != len(
                items
            ):
                raise ValueError("baseline items differ from frozen evaluation")
            if (
                record.get("errors", 0)
                or record.get("pending", 0)
                or record.get("missing", 0)
            ):
                raise ValueError("baseline is incomplete")
            if any(
                r.get("model") != config["arms"][arm]["model"]
                for r in record["results"]
            ):
                raise ValueError("baseline model differs")
            record["baseline_source_sha256"] = ref["sha256"]
            record["baseline_reused"] = True
            records.append((arm, track, record))
    return records


def load_runtime_credentials():
    """Load only the two explicitly provisioned provider credentials."""
    for provider, name in [
        ("fireworks", "FIREWORKS_API_KEY"),
        ("refuseless", "REFUSELESS_API_KEY"),
    ]:
        if os.environ.get(name):
            continue
        path = Path.home() / ".config" / provider / "credentials.env"
        if not path.exists():
            raise ValueError(f"missing credential: {name}")
        if path.stat().st_mode & 0o077:
            raise ValueError(f"credential file for {name} must be owner-only")
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key == name:
                os.environ[name] = value.strip().strip("\"'")
                break
        if not os.environ.get(name):
            raise ValueError(f"missing credential: {name}")


@contextmanager
def exclusive_controller(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a+") as lock:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another controller owns this experiment") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_pilot(config, run_dir, approval):
    # One live controller per checkout, even if callers choose different paths.
    root = Path(__file__).resolve().parents[2]
    with exclusive_controller(root / ".local/live-pilot.lock"):
        return _run_pilot(config, run_dir, approval)


def _run_pilot(config, run_dir, approval):
    """A/C baselines, one SFT pass, B, report and approved cleanup.

    One controller only. Completed requests may resume within the same session;
    unknown optimizer/session outcomes require reconciliation, never a new run.
    """
    if not approval.get("approval_quote") or not config.get("paid_execution"):
        raise ValueError("exact paid-plan approval is required")
    data = load_private_manifest(config["data_manifest"], config["data_sha256"])
    manifest = load_private_manifest(config["eval_manifest"], config["eval_sha256"])
    recovered = (
        verify_recovery_baselines(config, manifest)
        if config.get("baseline_files")
        else []
    )
    previous = {}
    for provider, reference in config.get("previous_ledgers", {}).items():
        raw = Path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("previous ledger hash differs from recovery plan")
        previous[provider] = RunState(Path(reference["path"]))
    # Complete local readiness validation BEFORE SDK session construction.
    train_sft(config, data, None, dry_run=True)
    if len(manifest["pilot"]) != 89:
        raise ValueError("approved pilot requires89 frozen evaluation items")
    if config.get("checkpoint_retention") != "delete_after_report":
        raise ValueError(
            "this runner supports approved delete-after-report policy only"
        )
    budgets = config["provider_budgets"]
    for provider, user_cap in [("fireworks", "50"), ("refuseless", "10")]:
        cap = Decimal(str(budgets[provider]))
        if not cap.is_finite() or cap <= 0 or cap > Decimal(user_cap):
            raise ValueError("provider budget exceeds user ceiling")
        if provider in previous:
            summary = previous[provider].summary()
            bound = (
                Decimal(summary["spent_usd"]) + Decimal(summary["reserved_usd"]) + cap
            )
            if bound > Decimal(str(config["cumulative_provider_caps"][provider])):
                raise ValueError("cumulative provider budget exceeded")
    if sum((Decimal(str(v)) for v in budgets.values()), Decimal(0)) > Decimal(
        str(config["budget_usd"])
    ):
        raise ValueError("provider budgets exceed approved overall budget")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    if (run_dir / "controller.json").exists():
        raise ValueError(
            "existing controller run requires session/optimizer reconciliation before restart"
        )
    load_runtime_credentials()
    deadline = min(
        datetime.fromisoformat(config["deadline_utc"].replace("Z", "+00:00")),
        datetime.now(UTC) + timedelta(minutes=60),
    )
    states = {
        provider: RunState(
            run_dir / (provider + "-events.jsonl"),
            budget_usd=budget,
            deadline_utc=deadline,
            request_ceiling=config["request_ceiling"],
            approved=True,
        )
        for provider, budget in budgets.items()
    }
    for state in states.values():
        state.assert_dispatchable()
    reporting_states = {
        name: LedgerGroup(state, previous[name]) if name in previous else state
        for name, state in states.items()
    }
    private_json(
        run_dir / "controller.json",
        {
            "run_id": config["run_id"],
            "status": "starting",
            "deadline": deadline.isoformat(),
            "approved_config_sha256": approval["config_sha256"],
        },
    )
    renderer = NativeRenderer.load(local_files_only=True)
    evaluations, callbacks = [], []
    session = checkpoint = None
    failure = None
    started = time.monotonic()

    def measure(name, arm, state, sampler=None):
        arm = {**arm, "name": name, "run_id": config["run_id"]}
        for track in ("capability", "behavior"):
            items = [
                i
                for i in manifest["pilot"]
                if (i["suite"] == "mmlu") == (track == "capability")
            ]
            result = evaluate(
                arm, {"items": items}, config["tracks"][track], state, sampler=sampler
            )
            result["items"] = items
            evaluations.append(result)
            private_json(run_dir / f"{name}-{track}.json", result)
            if result.get("errors") or result.get("pending") or result.get("missing"):
                raise ValueError(
                    "endpoint stopped; see private per-item status and reservations"
                )

    try:
        # Independent C remains a useful measured baseline if Fireworks capacity
        # later proves unavailable. No dataset or provider substitution occurs.
        if recovered:
            for arm, track, record in recovered:
                evaluations.append(record)
                private_json(run_dir / f"{arm}-{track}.json", record)
                states["fireworks"].record(
                    {
                        "type": "baseline_reused",
                        "arm": arm,
                        "track": track,
                        "source_sha256": record["baseline_source_sha256"],
                    }
                )
        else:
            measure("C", config["arms"]["C"], states["refuseless"])
        session = open_session(config, data, states["fireworks"])
        base_arm = {
            **config["arms"]["A"],
            "provider": "fireworks_checkpoint",
            "model": config["training"]["base_model"],
            "pricing": {"input_per_million": "0.60", "output_per_million": "3.60"},
            "supported_reasoning": ["off"],
            "maximum_attempts": 7,
            "billing_attempts_unobserved": True,
            "maximum_cost_multiplier": 7,
            "usage_accounts_for_all_attempts": False,
        }
        if not recovered:
            base_sampler = make_base_sampler(session["service"], renderer)
            callbacks.append(base_sampler)
            measure("A", base_arm, states["fireworks"], base_sampler)
        checkpoint = train_sft(
            config, data, states["fireworks"], dry_run=False, session=session
        )
        private_json(run_dir / "checkpoint.json", checkpoint.to_dict())
        trained_sampler = make_checkpoint_sampler(
            checkpoint, session["service"], renderer
        )
        callbacks.append(trained_sampler)
        tuned_arm = {**base_arm, "model": checkpoint.sampler_path}
        measure("B", tuned_arm, states["fireworks"], trained_sampler)
    except Exception as error:
        failure = type(error).__name__
        states["fireworks"].record(
            {"type": "controller_failure", "exception_type": failure}
        )
    finally:
        for callback in callbacks:
            try:
                callback.close()
            except Exception as error:
                states["fireworks"].record(
                    {
                        "type": "sampler_close_failed",
                        "exception_type": type(error).__name__,
                    }
                )
        readiness = {
            "Frozen and reviewed data": True,
            "Pinned native renderer": True,
            "C sampled": any(e["arm"] == "C" for e in evaluations),
            "A sampled": any(e["arm"] == "A" for e in evaluations),
            "Training checkpoint saved": checkpoint is not None,
            "B sampled": any(e["arm"] == "B" for e in evaluations),
        }
        artifacts = {}
        report_exported = False
        try:
            artifacts = build_report(
                CombinedLedger(reporting_states),
                run_dir / "report",
                evaluations=evaluations,
                readiness=readiness,
            )
            report_exported = True
        except Exception as error:
            failure = failure or "ReportExportFailed"
            states["fireworks"].record(
                {"type": "report_export_failed", "exception_type": type(error).__name__}
            )
            # Per-item snapshots and durable events already preserve the actual
            # result. Export a machine-readable fallback before cleanup.
            try:
                private_json(
                    run_dir / "report-fallback.json",
                    {
                        "status": "report_export_failed",
                        "evaluations": evaluations,
                        "accounting": pool_summary(reporting_states),
                    },
                )
                report_exported = True
            except OSError:
                pass
        if session is not None:
            try:
                if checkpoint and report_exported:
                    finalize_session(
                        checkpoint,
                        retain=False,
                        state=states["fireworks"],
                        service=session["service"],
                        report_exported=True,
                        retention_approved=True,
                    )
                else:
                    states["fireworks"].record(
                        {
                            "type": "failed_session_cleanup_intent",
                            "report_exported": report_exported,
                        }
                    )
                    session["service"].delete_training_session()
                    states["fireworks"].record({"type": "failed_session_deleted"})
            except Exception as error:
                failure = failure or "CleanupUnverified"
                states["fireworks"].record(
                    {
                        "type": "cleanup_unverified",
                        "exception_type": type(error).__name__,
                    }
                )
    result = {
        "run_id": config["run_id"],
        "status": "incomplete" if failure else "completed_generations",
        "exception_type": failure,
        "elapsed_seconds": time.monotonic() - started,
        "accounting": pool_summary(reporting_states),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "behavior_judging": "pending independent per-output review",
        "comparison_provenance": config["arms"]["C"].get("provenance"),
    }
    private_json(run_dir / "controller.json", result)
    return result
