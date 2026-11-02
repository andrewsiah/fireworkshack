"""No-network checks for one-submission waits and deadline handling."""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest

from fireworkshack.train import wait_for_training_phase


class State:
    def __init__(self, remaining=2):
        self.deadline = datetime.now(UTC) + timedelta(seconds=remaining)
        self.events = []

    def record(self, event):
        self.events.append(event)


class AsyncFuture:
    def __init__(self, duration, error=None):
        self.duration = duration
        self.error = error
        self.wait_calls = 0
        self.cancel_calls = 0

    async def result_async(self, timeout=None):
        self.wait_calls += 1
        await asyncio.sleep(self.duration)
        if self.error:
            raise self.error
        return "completed-once"

    def cancel(self):
        self.cancel_calls += 1


def test_single_async_future_survives_heartbeat_waits_and_completes():
    state = State()
    future = AsyncFuture(0.035)
    result = wait_for_training_phase(
        future,
        state,
        operation_id="run:train:0",
        phase="forward_backward",
        timeout_seconds=0.5,
        heartbeat_seconds=0.01,
    )
    assert result == "completed-once"
    assert future.wait_calls == 1 and future.cancel_calls == 0
    heartbeats = [e for e in state.events if e["type"] == "training_phase_heartbeat"]
    assert len(heartbeats) >= 2
    assert all(e["phase_id"] == "run:train:0:forward_backward" for e in state.events)
    assert state.events[-1]["type"] == "training_phase_complete"
    assert state.events[-1]["elapsed_seconds"] >= 0.03


def test_phase_wait_clamps_to_run_deadline_without_resubmission():
    state = State(remaining=0.035)
    future = AsyncFuture(10)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline|limit"):
        wait_for_training_phase(
            future,
            state,
            operation_id="run:train:0",
            phase="optim_step",
            timeout_seconds=600,
            heartbeat_seconds=0.01,
        )
    assert time.monotonic() - started < 0.3
    assert future.wait_calls == 1 and future.cancel_calls == 0
    assert state.events[-1]["type"] == "training_phase_wait_expired"
    assert state.events[-1]["remote_outcome"] == "unknown"


def test_completed_operation_timeout_is_not_reinterpreted_as_heartbeat():
    state = State()
    future = AsyncFuture(0, error=TimeoutError("underlying operation timeout"))
    with pytest.raises(TimeoutError, match="underlying operation"):
        wait_for_training_phase(
            future,
            state,
            operation_id="run:train:0",
            phase="optim_step",
            timeout_seconds=600,
            heartbeat_seconds=0.01,
        )
    assert future.wait_calls == 1
    assert not any(e["type"] == "training_phase_heartbeat" for e in state.events)
    assert state.events[-1]["type"] == "training_phase_failed"


def test_expired_deadline_does_not_start_wait():
    state = State(remaining=-1)
    future = AsyncFuture(0)
    with pytest.raises(TimeoutError):
        wait_for_training_phase(
            future, state, operation_id="x", phase="optim_step", timeout_seconds=600
        )
    assert future.wait_calls == 0


def test_sync_mock_future_reuses_same_object_after_wait_timeout():
    class Future:
        calls = 0

        def result(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError()
            return 7

    state = State()
    future = Future()
    assert (
        wait_for_training_phase(
            future,
            state,
            operation_id="x",
            phase="forward_backward",
            timeout_seconds=1,
            heartbeat_seconds=0.01,
        )
        == 7
    )
    assert future.calls == 2
    assert state.events[-1]["type"] == "training_phase_complete"


def test_sync_wait_deadline_is_not_caught_as_another_heartbeat():
    class Future:
        calls = 0

        def result(self, timeout=None):
            self.calls += 1
            if self.calls > 5:
                raise RuntimeError("synchronous wait failed to stop at deadline")
            time.sleep(timeout)
            raise TimeoutError()

    state = State(remaining=0.015)
    future = Future()
    with pytest.raises(TimeoutError, match="deadline|limit"):
        wait_for_training_phase(
            future,
            state,
            operation_id="x",
            phase="optim_step",
            timeout_seconds=600,
            heartbeat_seconds=0.01,
        )
    assert future.calls <= 2
