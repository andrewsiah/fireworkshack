from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fireworkshack.state import BudgetExceeded, PendingOperation, RunState


def state(tmp_path, **kwargs):
    return RunState(
        tmp_path / "events.jsonl",
        budget_usd=Decimal("0.03"),
        deadline_utc=datetime.now(UTC) + timedelta(minutes=5),
        request_ceiling=3,
        approved=True,
        **kwargs,
    )


def test_reservation_prevents_overspend_and_unknown_retry(tmp_path):
    run = state(tmp_path)
    run.reserve("first", Decimal("0.02"))
    with pytest.raises(BudgetExceeded):
        run.reserve("second", Decimal("0.02"))
    with pytest.raises(PendingOperation):
        state(tmp_path).reserve("first", Decimal("0.02"))
    run.settle("first", None)
    assert run.summary()["reserved_usd"] == "0.02"


def test_settlement_releases_only_known_usage(tmp_path):
    run = state(tmp_path)
    run.reserve("one", Decimal("0.02"))
    run.settle("one", Decimal("0.005"))
    assert run.summary()["spent_usd"] == "0.005"
    assert run.summary()["reserved_usd"] == "0"
    with pytest.raises(PendingOperation):
        run.reserve("one", Decimal("0.01"))


def test_limits_and_approval_checked_before_dispatch(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        RunState(tmp_path / "a", budget_usd=None, approved=False).reserve(
            "one", Decimal("0.01")
        )
    run = RunState(
        tmp_path / "b",
        budget_usd=Decimal("1"),
        approved=True,
        deadline_utc=datetime.now(UTC) - timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="deadline"):
        run.reserve("one", Decimal("0.01"))
    run = state(tmp_path)
    for i in range(3):
        run.reserve(str(i), Decimal("0"))
        run.settle(str(i), Decimal("0"))
    with pytest.raises(ValueError, match="request"):
        run.reserve("last", Decimal("0"))


def test_malformed_log_stops_resume(tmp_path):
    (tmp_path / "events.jsonl").write_text("{partial")
    with pytest.raises(ValueError, match="corrupt"):
        state(tmp_path).summary()


def test_concurrent_reservations_serialize(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def reserve(i):
        try:
            state(tmp_path).reserve(str(i), Decimal("0.02"))
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, [1, 2])) == 1


def test_real_cost_overrun_is_recorded_and_stops_future_dispatch(tmp_path):
    run = state(tmp_path)
    run.reserve("one", Decimal("0.01"))
    run.settle("one", Decimal("0.05"))
    assert run.summary()["spent_usd"] == "0.05"
    assert run.summary()["budget_breaches"]
    with pytest.raises(BudgetExceeded):
        run.reserve("two", Decimal("0"))


def test_deadline_is_rechecked_after_waiting_for_lock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    run = RunState(
        tmp_path / "events.jsonl",
        budget_usd=Decimal("1"),
        approved=True,
        deadline_utc=datetime.now(UTC) + timedelta(milliseconds=100),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        with run._locked():
            future = pool.submit(run.reserve, "one", Decimal("0.01"))
            Event().wait(0.2)
        with pytest.raises(ValueError, match="deadline"):
            future.result()
    assert run.summary()["operation_count"] == 0
