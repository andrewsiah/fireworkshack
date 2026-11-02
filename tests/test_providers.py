"""Synthetic transport tests: no credentials or live requests."""

import json
from decimal import Decimal

import httpx
import pytest

from fireworkshack.providers import (
    ProtocolError,
    build_request,
    generate,
    protocol_hash,
)


class State:
    def __init__(self, remaining="1"):
        self.remaining = Decimal(remaining)
        self.pending = set()
        self.events = []
        self.settled = []

    def assert_dispatchable(self):
        pass

    def reserve(self, operation_id, maximum_usd):
        if operation_id in self.pending:
            raise RuntimeError("pending operation needs reconciliation")
        if maximum_usd > self.remaining:
            raise RuntimeError("budget exceeded")
        self.pending.add(operation_id)
        self.remaining -= maximum_usd

    def record(self, event):
        self.events.append(event)

    def settle(self, operation_id, actual_usd):
        self.settled.append((operation_id, actual_usd))
        if actual_usd is not None:
            self.pending.remove(operation_id)


@pytest.fixture
def arm():
    return {
        "name": "C",
        "provider": "refuseless",
        "base_url": "https://api.refuseless.com/v1",
        "model": "deepseek-v4.1-flash-abliterated",
        "key_env": "REFUSELESS_API_KEY",
        "supported_reasoning": ["off", "low", "high", "max"],
        "pricing": {"input_per_million": "0.12", "output_per_million": "0.45"},
    }


@pytest.fixture
def item():
    return {
        "item_id": "synthetic-1",
        "suite": "mmlu",
        "prompt": "Synthetic question?",
        "expected_answer": "A",
    }


@pytest.fixture
def protocol():
    return {
        "reasoning": "off",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 16,
        "max_input_tokens": 1024,
        "mmlu_scoring": "generated_letter",
    }


def success():
    return {
        "model": "deepseek-v4.1-flash-abliterated",
        "choices": [
            {
                "message": {"content": "A", "reasoning_content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 1},
    }


def test_explicit_model_protocol_and_cost_reservation(arm, item, protocol):
    state = State()
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert state.pending, "reserve must precede dispatch"
        return httpx.Response(200, json=success())

    result = generate(
        arm,
        item,
        protocol,
        state,
        env={"REFUSELESS_API_KEY": "synthetic"},
        transport=httpx.MockTransport(handler),
    )
    assert calls[0]["model"] == arm["model"]
    assert calls[0]["reasoning_effort"] == "off"
    assert calls[0]["max_tokens"] == 16
    assert calls[0]["temperature"] == 0
    assert result["final_text"] == "A"
    assert result["protocol_hash"] == protocol_hash(protocol)
    assert result["actual_usd"] == "0.00000285"
    assert result["source"] == "synthetic"
    assert not state.pending


def test_dry_run_never_creates_client(arm, item, protocol, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("client created"))
    result = generate(arm, item, protocol, State(), dry_run=True)
    assert result["status"] == "planned"
    assert result["source"] == "planned"


def test_insufficient_budget_prevents_http(arm, item, protocol):
    with pytest.raises(RuntimeError, match="budget"):
        generate(
            arm,
            item,
            protocol,
            State("0.000001"),
            env={"REFUSELESS_API_KEY": "synthetic"},
            transport=httpx.MockTransport(lambda r: pytest.fail("dispatched")),
        )


def test_unknown_outcome_cannot_be_resent(arm, item, protocol):
    state, calls = State(), []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic timeout")

    result = generate(
        arm,
        item,
        protocol,
        state,
        env={"REFUSELESS_API_KEY": "synthetic"},
        transport=httpx.MockTransport(timeout),
    )
    assert result["status"] == "pending"
    with pytest.raises(RuntimeError, match="reconciliation"):
        generate(
            arm,
            item,
            protocol,
            state,
            env={"REFUSELESS_API_KEY": "synthetic"},
            transport=httpx.MockTransport(timeout),
        )
    assert len(calls) == 1


def test_missing_usage_keeps_cost_unsettled(arm, item, protocol):
    body = success()
    del body["usage"]
    state = State()
    result = generate(
        arm,
        item,
        protocol,
        state,
        env={"REFUSELESS_API_KEY": "synthetic"},
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    assert result["actual_usd"] is None
    assert state.pending


def test_unsupported_reasoning_and_logprob_are_rejected(arm, item, protocol):
    with pytest.raises(ProtocolError, match="reasoning"):
        build_request(arm, item, {**protocol, "reasoning": "medium"})
    with pytest.raises(ProtocolError, match="scoring"):
        build_request(arm, item, {**protocol, "mmlu_scoring": "first_token_logit"})


def test_fireworks_off_maps_to_documented_none(arm, item, protocol):
    body = build_request({**arm, "provider": "fireworks"}, item, protocol)
    assert body["reasoning_effort"] == "none"


def test_experiential_needs_pinned_route_and_no_fallback(arm, item, protocol):
    with pytest.raises(ProtocolError, match="route"):
        build_request({**arm, "provider": "experiential"}, item, protocol)
    body = build_request(
        {
            **arm,
            "provider": "experiential",
            "route_id": "fixed",
            "allow_fallbacks": False,
        },
        item,
        protocol,
    )
    assert body["gateway"]["routing"] == {"route_id": "fixed", "allow_fallbacks": False}


def test_contextual_input_preserved(arm, protocol):
    body = build_request(
        arm,
        {
            "item_id": "context",
            "suite": "harmbench",
            "prompt": "Synthetic task",
            "context": "Synthetic context",
        },
        protocol,
    )
    assert "Synthetic context" in body["messages"][-1]["content"]
    assert "Synthetic task" in body["messages"][-1]["content"]


def test_reasoning_only_is_incomplete_and_truncated(arm, item, protocol):
    body = success()
    body["choices"][0] = {
        "message": {"content": "", "reasoning_content": "Synthetic reasoning"},
        "finish_reason": "length",
    }
    result = generate(
        arm,
        item,
        protocol,
        State(),
        env={"REFUSELESS_API_KEY": "synthetic"},
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    assert result["reasoning_only"] is True
    assert result["truncated"] is True
    assert result["complete"] is False


def test_missing_key_reports_only_variable(arm, item, protocol):
    with pytest.raises(ValueError, match="REFUSELESS_API_KEY"):
        generate(arm, item, protocol, State(), env={})


def test_protocol_hash_is_independent_of_arm_and_key(protocol):
    assert protocol_hash(protocol) == protocol_hash(
        dict(reversed(list(protocol.items())))
    )


def test_remote_error_details_do_not_leak_credentials(arm, item, protocol):
    result = generate(
        arm,
        item,
        protocol,
        State(),
        env={"REFUSELESS_API_KEY": "fake-secret-never-log"},
        transport=httpx.MockTransport(
            lambda r: httpx.Response(401, json={"error": "fake-secret-never-log"})
        ),
    )
    assert result["error"] == "http_401"
    assert "fake-secret-never-log" not in str(result)


def test_internal_attempts_reserve_all_seven_and_keep_unobserved_usage_held(
    arm, item, protocol
):
    arm = {
        **arm,
        "provider": "fireworks_checkpoint",
        "maximum_attempts": 7,
        "billing_attempts_unobserved": True,
        "source": "synthetic",
    }
    state = State()
    one_attempt_maximum = (
        Decimal(1024) * Decimal("0.12") + Decimal(16) * Decimal("0.45")
    ) / Decimal(1000000)
    result = generate(arm, item, protocol, state, sampler=lambda body: success())
    assert Decimal(result["reserved_usd"]) == one_attempt_maximum * 7
    assert state.remaining == Decimal(1) - one_attempt_maximum * 7
    assert result["actual_usd"] is None
    assert result["final_attempt_cost_estimate"] == "0.00000285"
    assert result["cost_status"] == "unsettled_internal_attempt_usage_unobserved"
    assert state.pending
    assert state.settled[-1][1] is None


@pytest.mark.parametrize(
    "attempts", [0, -1, 8, True, False, 1.5, "7", float("nan"), float("inf")]
)
def test_invalid_attempt_bound_rejected_before_dispatch(arm, item, protocol, attempts):
    with pytest.raises(ProtocolError, match="maximum_attempts"):
        generate(
            {**arm, "maximum_attempts": attempts}, item, protocol, State(), dry_run=True
        )


@pytest.mark.parametrize("value", ["-0.1", "NaN", "Infinity"])
def test_invalid_rates_cannot_reduce_retry_reservation(arm, item, protocol, value):
    arm = {
        **arm,
        "pricing": {"input_per_million": value, "output_per_million": "0.45"},
        "maximum_attempts": 7,
        "billing_attempts_unobserved": True,
    }
    with pytest.raises(ProtocolError, match="pricing"):
        generate(arm, item, protocol, State(), dry_run=True)


def test_multiple_attempts_cannot_claim_final_attempt_covers_total(arm, item, protocol):
    with pytest.raises(ProtocolError, match="unobserved"):
        generate(
            {**arm, "maximum_attempts": 7, "billing_attempts_unobserved": False},
            item,
            protocol,
            State(),
            dry_run=True,
        )
