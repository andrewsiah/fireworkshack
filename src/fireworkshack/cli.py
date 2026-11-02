"""Offline by default; paid entry points require an exact private approval."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .config import load_config, preflight
from .report import build_report
from .state import RunState


def _json(path):
    return json.loads(Path(path).read_text())


def verify_approval(config_path: Path, approval_path: Path | None) -> dict:
    if approval_path is None:
        raise ValueError("exact paid-plan approval record is required")
    approval = _json(approval_path)
    expected = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if (
        approval.get("config_sha256") != expected
        or not approval.get("approved_at_utc")
        or not approval.get("approval_quote")
    ):
        raise ValueError("approval does not match the exact configuration")
    config = load_config(config_path)
    if not config.get("paid_execution"):
        raise ValueError("paid execution is disabled in approved configuration")
    return approval


def dry_plan(config):
    plan = {
        "mode": "dry-run",
        "network_calls": 0,
        "paid_requests": 0,
        "preflight": preflight(config, os.environ),
        "training": "data not ready",
        "evaluation": {"A": "not run", "B": "not run", "C": "not run"},
    }
    train_path = Path(config.get("data_manifest", "data/manifests/training.json"))
    eval_path = Path(config.get("eval_manifest", "data/manifests/evaluation.json"))
    if train_path.exists():
        manifest = _json(train_path)
        plan["training_counts"] = {
            split: len(manifest[split]) for split in ("train", "dev")
        }
        if manifest.get("ready"):
            from .train import train_sft

            try:
                plan["training"] = train_sft(config, manifest, None, dry_run=True)
            except ValueError as error:
                plan["training"] = {
                    "status": "unresolved_configuration",
                    "reason": str(error),
                }
    if eval_path.exists():
        manifest = _json(eval_path)
        plan["evaluation_counts"] = {
            "pilot": len(manifest["pilot"]),
            "final_reserved": len(manifest["final"]),
        }
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "run", "train", "evaluate", "report"]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pilot"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.execute and (args.offline or args.dry_run):
            raise ValueError("choose offline/dry-run or execute")
        if args.command == "preflight":
            if args.execute:
                raise ValueError(
                    "preflight is offline; read-only remote checks are separate"
                )
            result = preflight(config, os.environ)
        elif args.command == "report":
            state = RunState(args.run_dir / "events.jsonl")
            result = {
                key: str(path)
                for key, path in build_report(state, args.run_dir / "report").items()
            }
        elif not args.execute:
            result = dry_plan(config)
        else:
            if args.command != "run":
                raise ValueError(
                    "paid train/evaluate stages run together through run --execute"
                )
            approval = verify_approval(args.config, args.approval)
            from .runner import run_pilot

            result = run_pilot(config, args.run_dir, approval)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ValueError, OSError, KeyError) as error:
        # Exceptions from local validation contain field names, never key values.
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
