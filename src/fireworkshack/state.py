"""Durable reservations; uncertainty keeps money reserved and blocks redispatch."""

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


class BudgetExceeded(ValueError):
    pass


class PendingOperation(ValueError):
    pass


class RunState:
    def __init__(
        self,
        path: Path,
        *,
        budget_usd=None,
        deadline_utc=None,
        request_ceiling=300,
        approved=False,
    ):
        self.path = Path(path)
        self.budget = Decimal(str(budget_usd)) if budget_usd is not None else None
        self.deadline = (
            datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
            if isinstance(deadline_utc, str)
            else deadline_utc
        )
        self.request_ceiling = request_ceiling
        self.approved = approved

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.path.open("a+", encoding="utf8") as file:
            os.chmod(self.path, 0o600)
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                file.seek(0)
                try:
                    events = [json.loads(line) for line in file if line.strip()]
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        "corrupt event log; reconcile before resume"
                    ) from exc
                yield file, events
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    @staticmethod
    def _append(file, event):
        file.seek(0, os.SEEK_END)
        file.write(
            json.dumps({"utc": datetime.now(UTC).isoformat(), **event}, default=str)
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())

    @staticmethod
    def _accounting(events):
        operations = {}
        for e in events:
            if e.get("type") == "reserve":
                operations[e["operation_id"]] = {
                    "maximum": Decimal(e["maximum_usd"]),
                    "actual": None,
                }
            elif e.get("type") == "settle":
                op = operations.get(e["operation_id"])
                if op is None:
                    raise ValueError("corrupt settlement without reservation")
                op["actual"] = Decimal(e["actual_usd"])
        spent = sum(
            (v["actual"] for v in operations.values() if v["actual"] is not None),
            Decimal(0),
        )
        held = sum(
            (v["maximum"] for v in operations.values() if v["actual"] is None),
            Decimal(0),
        )
        return operations, spent, held

    def assert_dispatchable(self):
        if not self.approved or self.budget is None:
            raise ValueError("resolved paid plan approval and budget required")
        if not self.budget.is_finite() or self.budget <= 0:
            raise ValueError("positive finite budget required")
        if (
            self.deadline is None
            or self.deadline.tzinfo is None
            or datetime.now(UTC) >= self.deadline
        ):
            raise ValueError("deadline absent or expired")
        with self._locked() as (_, events):
            _, spent, held = self._accounting(events)
            if (
                any(e.get("type") == "budget_breach" for e in events)
                or spent + held > self.budget
            ):
                raise BudgetExceeded(
                    "prior usage exceeded its reservation; reconcile before dispatch"
                )

    def reserve(self, operation_id: str, maximum_usd: Decimal):
        self.assert_dispatchable()
        amount = Decimal(str(maximum_usd))
        if not amount.is_finite() or amount < 0:
            raise ValueError("reservation must be finite and nonnegative")
        with self._locked() as (file, events):
            if datetime.now(UTC) >= self.deadline:
                raise ValueError("deadline expired while waiting for ledger lock")
            operations, spent, held = self._accounting(events)
            if operation_id in operations:
                raise PendingOperation(
                    "operation already reserved or completed; reconcile instead of redispatch"
                )
            if len(operations) >= self.request_ceiling:
                raise ValueError("request ceiling reached")
            if spent + held + amount > self.budget:
                raise BudgetExceeded("reservation would exceed approved budget")
            self._append(
                file,
                {
                    "type": "reserve",
                    "operation_id": operation_id,
                    "maximum_usd": str(amount),
                },
            )

    def record(self, event: dict):
        if event.get("type") in {"reserve", "settle"}:
            raise ValueError("use reservation/settlement methods for accounting events")
        with self._locked() as (file, _):
            self._append(file, event)

    def settle(self, operation_id: str, actual_usd):
        with self._locked() as (file, events):
            operations, _, _ = self._accounting(events)
            if operation_id not in operations:
                raise ValueError("operation has no reservation")
            if operations[operation_id]["actual"] is not None:
                raise PendingOperation("operation already settled")
            if actual_usd is None:
                self._append(
                    file, {"type": "usage_uncertain", "operation_id": operation_id}
                )
                return
            amount = Decimal(str(actual_usd))
            if not amount.is_finite() or amount < 0:
                raise ValueError("actual cost must be finite and nonnegative")
            self._append(
                file,
                {
                    "type": "settle",
                    "operation_id": operation_id,
                    "actual_usd": str(amount),
                },
            )
            if amount > operations[operation_id]["maximum"]:
                self._append(
                    file,
                    {
                        "type": "budget_breach",
                        "operation_id": operation_id,
                        "actual_usd": str(amount),
                        "reserved_usd": str(operations[operation_id]["maximum"]),
                    },
                )

    def events(self):
        with self._locked() as (_, events):
            return events

    def summary(self):
        with self._locked() as (_, events):
            operations, spent, held = self._accounting(events)
            return {
                "spent_usd": str(spent),
                "reserved_usd": str(held),
                "operation_count": len(operations),
                "budget_breaches": [
                    e for e in events if e.get("type") == "budget_breach"
                ],
                "pending_operations": [
                    k for k, v in operations.items() if v["actual"] is None
                ],
            }
