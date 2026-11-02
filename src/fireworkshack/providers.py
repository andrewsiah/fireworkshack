"""Explicit, bounded hosted generation. Results belong in private run storage.

The adapter never retries a request with an uncertain outcome. A public catalog
listing is not protocol verification; callers supply the verified capabilities.
REST contracts: Fireworks post-chatcompletions and Refuseless chat-completions /
reasoning docs, checked 2026-11-02. No client is constructed for dry runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import httpx


class ProtocolError(ValueError):
    """The exact requested protocol cannot be honored by this adapter."""


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(protocol), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def operation_id(
    arm: Mapping[str, Any], item: Mapping[str, Any], protocol: Mapping[str, Any]
) -> str:
    """Bind resume identity to run, provider, route, exact item and protocol."""
    identity = {
        "run_id": arm.get("run_id"),
        "arm": arm.get("name"),
        "model": arm.get("model"),
        "provider": arm.get("provider"),
        "base_url": arm.get("base_url"),
        "route_id": arm.get("route_id"),
        "item": dict(item),
        "protocol": dict(protocol),
    }
    return "generate-" + protocol_hash(identity)


def format_messages(item: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[dict]:
    """Keep the complete contextual input; never turn context into training data."""
    prompt = str(item.get("prompt", ""))
    if not prompt.strip():
        raise ProtocolError("item prompt is empty")
    context = item.get("context")
    if context:
        prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
    if item.get("suite", "").lower() == "mmlu":
        choices = item.get("choices")
        if choices:
            if len(choices) != 4:
                raise ProtocolError("MMLU requires four choices")
            prompt += "\n" + "\n".join(
                f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
            )
        prompt += "\nReply with exactly one letter: A, B, C, or D."
    messages = []
    if protocol.get("system_message"):
        messages.append({"role": "system", "content": str(protocol["system_message"])})
    messages.append({"role": "user", "content": prompt})
    return messages


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{name} must be a positive integer")
    return value


def build_request(
    arm: Mapping[str, Any], item: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict:
    required = {
        "reasoning",
        "temperature",
        "top_p",
        "max_tokens",
        "max_input_tokens",
        "mmlu_scoring",
    }
    if required - protocol.keys():
        raise ProtocolError(
            "explicit protocol fields missing: "
            + ", ".join(sorted(required - protocol.keys()))
        )
    if protocol["mmlu_scoring"] != "generated_letter":
        raise ProtocolError(
            "only generated_letter MMLU scoring is implemented; logit scoring must not be mixed"
        )
    effort = protocol["reasoning"]
    if effort not in arm.get("supported_reasoning", []):
        raise ProtocolError("reasoning setting has not been verified for this arm")
    for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = protocol[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ProtocolError(f"invalid {name}")
    if protocol["top_p"] == 0:
        raise ProtocolError("top_p must be greater than zero")
    _positive_int(protocol["max_tokens"], "max_tokens")
    _positive_int(protocol["max_input_tokens"], "max_input_tokens")
    if protocol.get("logprobs") or protocol.get("top_logprobs"):
        raise ProtocolError("logprob scoring is not implemented")
    provider = arm.get("provider")
    if provider not in {
        "fireworks",
        "refuseless",
        "experiential",
        "fireworks_checkpoint",
    }:
        raise ProtocolError("unsupported provider")
    if not arm.get("model"):
        raise ProtocolError("explicit model is required")
    messages = format_messages(item, protocol)
    # Conservative text-only byte bound with a large chat-template allowance.
    # A tokenizer/renderer check remains mandatory in the live preflight.
    byte_bound = 512 + sum(len(m["content"].encode("utf-8")) for m in messages)
    if byte_bound > protocol["max_input_tokens"]:
        raise ProtocolError("prompt exceeds conservative input-token reservation bound")
    if (
        arm.get("context_length")
        and protocol["max_input_tokens"] + protocol["max_tokens"]
        > arm["context_length"]
    ):
        raise ProtocolError("reserved input and output exceed model context")
    body = {
        "model": arm["model"],
        "messages": messages,
        "temperature": protocol["temperature"],
        "top_p": protocol["top_p"],
        "max_tokens": protocol["max_tokens"],
        "stream": False,
        "reasoning_effort": "none"
        if provider == "fireworks" and effort == "off"
        else effort,
    }
    if provider == "experiential":
        if not arm.get("route_id") or arm.get("allow_fallbacks") is not False:
            raise ProtocolError(
                "Experiential requires a pinned route and disabled fallbacks"
            )
        body["gateway"] = {
            "routing": {"route_id": arm["route_id"], "allow_fallbacks": False}
        }
    return body


def _rates(arm: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    try:
        rates = tuple(
            Decimal(str(arm["pricing"][key]))
            for key in ("input_per_million", "output_per_million")
        )
    except (KeyError, ValueError, ArithmeticError) as exc:
        raise ProtocolError("verified input/output pricing is required") from exc
    if any(not value.is_finite() or value < 0 for value in rates):
        raise ProtocolError("pricing must be finite and nonnegative")
    return rates


def _cost(tokens_in: int, tokens_out: int, rates: tuple[Decimal, Decimal]) -> Decimal:
    # Charge all input as uncached for the reservation and usage estimate.
    return (Decimal(tokens_in) * rates[0] + Decimal(tokens_out) * rates[1]) / Decimal(
        1_000_000
    )


def _text_fields(message: Mapping[str, Any]) -> tuple[str, str]:
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(text, str) or not isinstance(reasoning, str):
        raise ProtocolError("non-text response is unsupported")
    if "<think>" in text:
        match = re.fullmatch(r"\s*<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
        if match:
            reasoning = "\n".join(filter(None, (reasoning, match[1])))
            text = match[2]
        elif text.lstrip().startswith("<think>"):
            reasoning = "\n".join(
                filter(None, (reasoning, text.split("<think>", 1)[1]))
            )
            text = ""
    return text, reasoning


def generate(
    arm: Mapping[str, Any],
    item: Mapping[str, Any],
    protocol: Mapping[str, Any],
    state: Any,
    *,
    env: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    dry_run: bool = False,
    sampler: Any = None,
) -> dict:
    """Generate once, after reservation, preserving unknown billing outcomes.

    `sampler` may be supplied for an in-session checkpoint and must accept the
    request dict and return the same normalized OpenAI response shape. It must
    honor the verified rendering/decoding contract; this module does not invent
    an HTTP model identifier from a training checkpoint handle.
    """
    body = build_request(arm, item, protocol)
    rates = _rates(arm)
    attempts = _positive_int(arm.get("maximum_attempts", 1), "maximum_attempts")
    if attempts > 7:
        raise ProtocolError("maximum_attempts exceeds the supported bound of 7")
    attempts_unobserved = arm.get("billing_attempts_unobserved", attempts > 1)
    if not isinstance(attempts_unobserved, bool):
        raise ProtocolError("billing_attempts_unobserved must be boolean")
    if attempts > 1 and not attempts_unobserved:
        raise ProtocolError(
            "multiple-attempt billing must remain unobserved until reconciled"
        )
    maximum = (
        _cost(protocol["max_input_tokens"], protocol["max_tokens"], rates) * attempts
    )
    request_id = operation_id(arm, item, protocol)
    result = {
        "operation_id": request_id,
        "item_id": item["item_id"],
        "suite": item.get("suite"),
        "arm": arm.get("name"),
        "model": arm["model"],
        "provider": arm["provider"],
        "protocol_hash": protocol_hash(protocol),
        "status": "planned",
        "source": "planned"
        if dry_run
        else (
            "synthetic"
            if transport is not None or arm.get("source") == "synthetic"
            else "measured"
        ),
        "final_text": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "actual_usd": None,
        "reserved_usd": str(maximum),
        "maximum_attempts": attempts,
        "billing_attempts_unobserved": attempts_unobserved,
        "final_attempt_cost_estimate": None,
        "latency_seconds": 0.0,
        "error": None,
        "complete": False,
        "truncated": False,
        "reasoning_only": False,
        "backend": arm.get("route_id"),
        "mmlu_scoring": protocol["mmlu_scoring"],
    }
    if dry_run:
        return result
    if arm["provider"] == "fireworks_checkpoint":
        if sampler is None:
            raise ProtocolError(
                "checkpoint sampling requires an explicit in-session sampler"
            )
    else:
        allowed = {
            "refuseless": "https://api.refuseless.com/v1",
            "fireworks": "https://api.fireworks.ai/inference/v1",
            "experiential": "https://api.experientiallabs.ai/v1",
        }
        if arm.get("base_url", "").rstrip("/") != allowed[arm["provider"]]:
            raise ProtocolError("base URL does not match the selected provider")
        key_name = arm.get("key_env")
        key = (os.environ if env is None else env).get(key_name or "", "")
        if not key:
            raise ValueError(f"missing credential: {key_name or 'key_env'}")
    state.assert_dispatchable()
    state.reserve(request_id, maximum)
    state.record(
        {
            "type": "dispatch_intent",
            "operation_id": request_id,
            "item_id": item["item_id"],
            "arm": arm.get("name"),
            "protocol_hash": result["protocol_hash"],
        }
    )
    start = time.monotonic()
    try:
        if arm["provider"] == "fireworks_checkpoint":
            response_data = sampler(body)
        else:
            with httpx.Client(
                transport=transport, timeout=30.0, follow_redirects=False
            ) as client:
                response = client.post(
                    arm["base_url"].rstrip("/") + "/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                )
            if response.status_code != 200:
                result.update(status="error", error=f"http_{response.status_code}")
                state.settle(request_id, None)
                return _record_result(result, state, start)
            response_data = response.json()
        choice = response_data["choices"][0]
        text, reasoning = _text_fields(choice["message"])
        finish = choice.get("finish_reason")
        result.update(
            status="completed",
            final_text=text,
            reasoning=reasoning,
            finish_reason=finish,
            truncated=finish in {"length", "max_tokens", "max_completion_tokens"},
            reasoning_only=bool(reasoning.strip()) and not bool(text.strip()),
            complete=bool(text.strip())
            and finish == "stop"
            and not choice["message"].get("tool_calls"),
            backend=response_data.get("system_fingerprint")
            or response_data.get("backend")
            or arm.get("route_id"),
            response_model=response_data.get("model"),
        )
        usage = response_data.get("usage")
        valid_usage = isinstance(usage, dict) and all(
            isinstance(usage.get(k), int)
            and not isinstance(usage[k], bool)
            and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        result["usage"] = usage if isinstance(usage, dict) else None
        if (
            valid_usage
            and usage["prompt_tokens"] <= protocol["max_input_tokens"]
            and usage["completion_tokens"] <= protocol["max_tokens"]
        ):
            actual = _cost(usage["prompt_tokens"], usage["completion_tokens"], rates)
            if attempts_unobserved:
                # SDK retries can incur charges absent from its final response.
                # Final-attempt token counts cannot release a whole-call hold.
                result["final_attempt_cost_estimate"] = str(actual)
                result["cost_status"] = "unsettled_internal_attempt_usage_unobserved"
                state.settle(request_id, None)
            else:
                result["actual_usd"] = str(actual)
                result["cost_status"] = "usage_estimate_uncached"
                state.settle(request_id, actual)
        else:
            result["cost_status"] = "unsettled_usage_missing_or_outside_bounds"
            state.settle(request_id, None)
    except (httpx.HTTPError, TimeoutError):
        result.update(status="pending", error="transport_outcome_unknown")
        state.settle(request_id, None)
    except (ValueError, KeyError, IndexError, TypeError):
        result.update(status="pending", error="response_contract_unknown")
        state.settle(request_id, None)
    return _record_result(result, state, start)


def _record_result(result: dict, state: Any, start: float) -> dict:
    result["latency_seconds"] = time.monotonic() - start
    state.record(
        {"type": "generation", "operation_id": result["operation_id"], "result": result}
    )
    return result
