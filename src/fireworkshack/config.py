"""Offline configuration checks. No provider client is constructed here."""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

BASE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"


def validate_config(config: dict) -> dict:
    budget = config.get("budget_usd")
    if budget is not None:
        try:
            amount = Decimal(str(budget))
            if not amount.is_finite() or amount <= 0:
                raise ValueError("budget_usd must be positive and finite")
        except InvalidOperation as exc:
            raise ValueError("invalid budget_usd") from exc
    if type(config.get("request_ceiling")) is not int or config["request_ceiling"] <= 0:
        raise ValueError("request_ceiling must be positive")
    if type(config.get("max_retries")) is not int or config["max_retries"] < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    deadline = config.get("deadline_utc")
    if deadline:
        value = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        if value.tzinfo is None:
            raise ValueError("deadline_utc must contain a timezone")
    if config.get("paid_execution") and (budget is None or deadline is None):
        raise ValueError("paid execution needs a budget and deadline")
    if config.get("training", {}).get("base_model") != BASE_MODEL:
        raise ValueError("unexpected training base model")
    return config


def load_config(path: Path) -> dict:
    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return validate_config(config)


def preflight(config: dict, env: Mapping[str, str]) -> dict:
    validate_config(config)
    missing = [
        name
        for name in ("FIREWORKS_API_KEY", "REFUSELESS_API_KEY")
        if not env.get(name)
    ]
    return {
        "mode": "offline",
        "network_calls": 0,
        "paid_ready": False,
        "missing_credentials": missing,
        "checks": {
            "configuration": "verified",
            "credential_presence": "failed" if missing else "verified",
            "fireworks_capacity": "unverified",
            "comparison_checkpoint_revision": "unverified",
            "renderer": "unverified",
            "dataset_review": "unverified",
            "resolved_paid_plan_approval": "unverified",
        },
        "budget_usd": config.get("budget_usd"),
        "paid_execution_enabled": bool(config.get("paid_execution")),
    }
