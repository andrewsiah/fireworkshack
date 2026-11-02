"""Bounded Fireworks serverless SFT using the supported SDK, with offline defaults.

The pinned native DeepSeek encoder is used only for text, single-turn SFT.
Training and sampler checkpoints are deliberately distinct. No API client is
constructed until all local validation and the caller's budget gate pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

BASE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
TOKENIZER_REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
TOKENIZER_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ENCODER_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
TRAIN_USD_PER_MILLION = Decimal("4.98")
SDK_MAX_SAMPLE_ATTEMPTS = 7


class TrainingNotReady(ValueError):
    """Local data/protocol prerequisites are not satisfied."""


def wait_for_training_phase(
    future: Any,
    state: Any,
    *,
    operation_id: str,
    phase: str,
    timeout_seconds: float,
    heartbeat_seconds: float = 10,
) -> Any:
    """Wait on one submitted future; heartbeat timeouts never resubmit it.

    The production SDK exposes result_async(), whose underlying operation is
    shielded from caller timeouts. One local task observes that same future;
    asyncio.wait leaves it running between heartbeat intervals. A deadline
    expiry is an unknown remote outcome, not a failed or canceled update.
    Synchronous injected test doubles retain their existing result() contract.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 600
        or not math.isfinite(heartbeat_seconds)
        or heartbeat_seconds <= 0
    ):
        raise TrainingNotReady(
            "Training phase wait must be positive and at most600 seconds"
        )
    started = time.monotonic()
    limit = started + timeout_seconds
    deadline = getattr(state, "deadline", None)
    if deadline is not None:
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            raise TrainingNotReady("Run deadline must include a timezone")
        limit = min(limit, started + (deadline - datetime.now(UTC)).total_seconds())
    identity = {
        "operation_id": operation_id,
        "phase": phase,
        "phase_id": f"{operation_id}:{phase}",
    }

    def record(kind: str, **extra: Any) -> None:
        state.record(
            {
                "type": kind,
                **identity,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                **extra,
            }
        )

    def remaining() -> float:
        left = limit - time.monotonic()
        if left <= 0:
            record("training_phase_wait_expired", remote_outcome="unknown")
            raise TimeoutError(
                "Training phase wait limit or run deadline reached; remote outcome unknown"
            )
        return left

    remaining()
    record(
        "training_phase_started",
        timeout_seconds=timeout_seconds,
        effective_wait_seconds=max(0.0, limit - started),
        server_request_id=None,
    )

    async def observe_async() -> Any:
        task = asyncio.create_task(future.result_async())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {task}, timeout=min(heartbeat_seconds, remaining())
                )
                if done:
                    # A completed operation raising TimeoutError is a terminal
                    # operation error, not another local heartbeat interval.
                    return task.result()
                record("training_phase_heartbeat", remote_outcome="pending")
        finally:
            if not task.done():
                task.cancel()  # local observer only; never future.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    try:
        if callable(getattr(future, "result_async", None)):
            result = asyncio.run(observe_async())
        else:
            while True:
                interval = min(heartbeat_seconds, remaining())
                try:
                    result = future.result(timeout=interval)
                    break
                except TimeoutError:
                    # Public concurrent-future access is available on the SDK
                    # wrapper; distinguish an actual completed exception where
                    # supported by an injected synchronous future as well.
                    underlying = getattr(future, "future", None)
                    if callable(underlying) and underlying().done():
                        raise
                    record("training_phase_heartbeat", remote_outcome="pending")
    except Exception as error:
        if not isinstance(error, TimeoutError) or time.monotonic() < limit:
            record(
                "training_phase_failed",
                exception_type=type(error).__name__,
                remote_outcome="unknown",
            )
        raise
    record("training_phase_complete")
    return result


@dataclass(frozen=True)
class CheckpointRef:
    base_model: str
    session_id: str
    run_id: str
    sampler_path: str
    training_path: str
    step: int
    tokenizer_revision: str
    data_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


class NativeRenderer:
    """Reviewed, hash-checked native encoder and pinned Rust tokenizer.

    Loading downloads tokenizer/encoder assets only, never model weights. This
    is explicit so dry-run training does not download anything implicitly.
    """

    revision = TOKENIZER_REVISION

    def __init__(self, tokenizer: Any, encoder: Any):
        self.tokenizer = tokenizer
        self.encoder = encoder

    @classmethod
    def load(cls, *, local_files_only: bool = False) -> NativeRenderer:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        paths = {}
        for filename, expected in [
            ("encoding/encoding.py", ENCODER_SHA256),
            ("tokenizer.json", TOKENIZER_SHA256),
        ]:
            path = Path(
                hf_hub_download(
                    TOKENIZER_REPOSITORY,
                    filename,
                    revision=TOKENIZER_REVISION,
                    local_files_only=local_files_only,
                )
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise TrainingNotReady("Pinned tokenizer/encoder checksum mismatch")
            paths[filename] = path
        spec = importlib.util.spec_from_file_location(
            "fireworkshack_pinned_deepseek_encoder", paths["encoding/encoding.py"]
        )
        if spec is None or spec.loader is None:
            raise TrainingNotReady("Cannot load pinned encoder")
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)
        return cls(Tokenizer.from_file(str(paths["tokenizer.json"])), encoder)

    def _tokens(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def render_supervised(self, prompt: str, response: str) -> dict:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(response, str)
            or not response.strip()
        ):
            raise TrainingNotReady("SFT requires nonempty text prompt and response")
        messages = [{"role": "user", "content": prompt}]
        prefix = self.encoder.encode_messages(messages, thinking_mode="chat")
        full = self.encoder.encode_messages(
            messages + [{"role": "assistant", "content": response}],
            thinking_mode="chat",
        )
        prefix_ids, token_ids = self._tokens(prefix), self._tokens(full)
        if token_ids[: len(prefix_ids)] != prefix_ids or len(token_ids) <= len(
            prefix_ids
        ):
            raise TrainingNotReady(
                "Native assistant prefix is not a token-aligned prefix"
            )
        return {
            "token_ids": token_ids,
            "weights": [0.0] * len(prefix_ids)
            + [1.0] * (len(token_ids) - len(prefix_ids)),
            "rendered_tokens": len(token_ids),
            "renderer_revision": self.revision,
        }

    def render_prompt(
        self, messages: list[dict], *, reasoning: str = "off"
    ) -> list[int]:
        if reasoning not in {"off", "low", "high", "max"}:
            raise TrainingNotReady("Unsupported reasoning mode")
        kwargs = {"thinking_mode": "chat" if reasoning == "off" else "thinking"}
        if reasoning != "off":
            kwargs["reasoning_effort"] = reasoning
        return self._tokens(self.encoder.encode_messages(messages, **kwargs))

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


def prepare_training_data(manifest: dict, renderer: NativeRenderer) -> dict:
    prepared = copy.deepcopy(manifest)
    for split in ["train", "dev"]:
        for row in prepared[split]:
            row.update(renderer.render_supervised(row["prompt"], row["response"]))
    prepared["renderer_revision"] = renderer.revision
    prepared["train_tokens"] = sum(len(x["token_ids"]) for x in prepared["train"])
    prepared["max_rendered_tokens"] = max(
        len(x["token_ids"]) for split in ["train", "dev"] for x in prepared[split]
    )
    prepared["ready"] = (
        len(prepared["train"]) == 64
        and len(prepared["dev"]) == 16
        and prepared["max_rendered_tokens"] <= 1024
        and prepared["train_tokens"] <= 65536
        and all(
            x.get("review_approved") is True
            for split in ["train", "dev"]
            for x in prepared[split]
        )
    )
    prepared["manifest_hash"] = hashlib.sha256(
        json.dumps(prepared, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return prepared


def _settings(config: dict) -> dict:
    raw = dict(config.get("training", {}))
    aliases = {
        "rank": "lora_rank",
        "max_sequence_length": "max_seq_len",
        "max_training_tokens": "max_train_tokens",
    }
    for alias, target in aliases.items():
        if target not in raw and alias in raw:
            raw[target] = raw[alias]
    defaults = {
        "base_model": BASE_MODEL,
        "lora_rank": 8,
        "epochs": 1,
        "learning_rate": 0.0001,
        "max_seq_len": 1024,
        "max_train_tokens": 65536,
        "batch_size": 8,
        "train_usd_per_million": "4.98",
        "alpha": 32,
        "seed": 42,
        "timeout_seconds": 120,
    }
    return {**defaults, **raw}


def _validate(config: dict, data: dict) -> tuple[dict, list[dict], int]:
    c = _settings(config)
    expected = {
        "base_model": BASE_MODEL,
        "lora_rank": 8,
        "epochs": 1,
        "learning_rate": 0.0001,
    }
    if any(c[k] != v for k, v in expected.items()):
        raise TrainingNotReady("Settings differ from the reviewed single-pass pilot")
    if (
        not 1 <= c["max_seq_len"] <= 1024
        or not 1 <= c["max_train_tokens"] <= 65536
        or not 1 <= c["batch_size"] <= 64
    ):
        raise TrainingNotReady("Training token/batch bounds exceeded")
    if (
        not c.get("renderer_verified")
        or c.get("tokenizer_revision") != TOKENIZER_REVISION
    ):
        raise TrainingNotReady("Pinned renderer/tokenizer verification required")
    if not data.get("ready") or data.get("renderer_revision") != TOKENIZER_REVISION:
        raise TrainingNotReady("Reviewed, rendered data manifest is not ready")
    rows = data.get("train", [])
    if len(rows) != 64 or len(data.get("dev", [])) != 16:
        raise TrainingNotReady("Pilot requires64 train and16 separate development rows")
    total = 0
    for row in rows:
        tokens, weights = row.get("token_ids"), row.get("weights")
        if not tokens or not weights or len(tokens) != len(weights) or len(tokens) < 2:
            raise TrainingNotReady(
                "Missing/misaligned rendered tokens and assistant mask"
            )
        if len(tokens) > c["max_seq_len"] or not any(float(w) > 0 for w in weights[1:]):
            raise TrainingNotReady("Oversized or unsupervised training row")
        if any(not isinstance(t, int) or t < 0 for t in tokens) or any(
            float(w) not in {0.0, 1.0} for w in weights
        ):
            raise TrainingNotReady("Invalid tokens or loss mask")
        if float(weights[0]) != 0.0 or list(map(float, weights)) != sorted(
            map(float, weights)
        ):
            raise TrainingNotReady(
                "Expected contiguous assistant-only mask after masked prompt"
            )
        total += len(tokens)
    if total > c["max_train_tokens"]:
        raise TrainingNotReady("Total training token ceiling exceeded")
    if Decimal(str(c["train_usd_per_million"])) != TRAIN_USD_PER_MILLION:
        raise TrainingNotReady(
            "Training price differs from the verified rate; resolve and approve again"
        )
    timeout = c["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 600
    ):
        raise TrainingNotReady(
            "Training timeout_seconds must be positive and at most600"
        )
    return c, rows, total


def train_sft(
    config: dict,
    data: dict,
    state: Any,
    *,
    dry_run: bool = True,
    service_factory: Any = None,
    session: dict | None = None,
) -> dict | CheckpointRef:
    c, rows, total = _validate(config, data)
    plan = {
        "status": "dry_run",
        "base_model": BASE_MODEL,
        "rows": len(rows),
        "train_tokens": total,
        "steps": math.ceil(len(rows) / c["batch_size"]),
        "tokenizer_revision": TOKENIZER_REVISION,
        "maximum_training_usd": str(
            Decimal(total) * TRAIN_USD_PER_MILLION / Decimal(1_000_000)
        ),
        "note": "Training only; sampling, judging and unsettled usage accounted separately.",
    }
    if dry_run:
        return plan
    state.assert_dispatchable()
    from fireworks.training import sdk

    if service_factory is None:
        service_factory = sdk.FiretitanServiceClient
    run = str(config.get("run_id", "pilot"))
    # Reserve the first batch before creating a session, so a resumed uncertain
    # first step cannot accidentally create another session.
    batches = [
        rows[i : i + c["batch_size"]] for i in range(0, len(rows), c["batch_size"])
    ]

    def reserve_batch(index: int, batch: list[dict]) -> str:
        operation = f"{run}:train:{index}"
        reserve = (
            Decimal(sum(len(x["token_ids"]) for x in batch))
            * TRAIN_USD_PER_MILLION
            / Decimal(1_000_000)
        )
        state.reserve(operation, reserve)
        return operation

    first_op = reserve_batch(0, batches[0])
    active = session or open_session(
        config, data, state, service_factory=service_factory
    )
    service, client = active["service"], active["client"]
    adam = sdk.AdamParams(
        learning_rate=c["learning_rate"],
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )
    for index, batch in enumerate(batches):
        state.assert_dispatchable()
        operation = first_op if index == 0 else reserve_batch(index, batch)
        datums = [
            sdk.Datum(
                model_input=sdk.ModelInput.from_ints(x["token_ids"][:-1]),
                loss_fn_inputs={
                    "target_tokens": x["token_ids"][1:],
                    "weights": x["weights"][1:],
                },
            )
            for x in batch
        ]
        state.record(
            {
                "type": "training_step_intent",
                "operation_id": operation,
                "step": index + 1,
                "rows": [x["item_id"] for x in batch],
            }
        )
        # No retry on an unknown outcome. The reservation/intent survives and
        # must be reconciled against session/optimizer state before resuming.
        state.assert_dispatchable()
        fb = wait_for_training_phase(
            client.forward_backward(datums, "cross_entropy"),
            state,
            operation_id=operation,
            phase="forward_backward",
            timeout_seconds=c["timeout_seconds"],
        )
        state.assert_dispatchable()
        state.record(
            {
                "type": "training_phase_intent",
                "operation_id": operation,
                "phase_id": f"{operation}:optim_step",
                "phase": "optim_step",
            }
        )
        opt = wait_for_training_phase(
            client.optim_step(adam, emit_grad_norm_metrics=True),
            state,
            operation_id=operation,
            phase="optim_step",
            timeout_seconds=c["timeout_seconds"],
        )
        state.record(
            {
                "type": "training_step_complete",
                "operation_id": operation,
                "step": index + 1,
                "metrics": {
                    **dict(getattr(fb, "metrics", {}) or {}),
                    **dict(getattr(opt, "metrics", {}) or {}),
                },
            }
        )
        # The update completed, but billed USD is unknown until reconciled.
        state.settle(operation, None)
    state.record({"type": "checkpoint_intent", "name": "pilot-final"})
    state.assert_dispatchable()
    training_path = wait_for_training_phase(
        client.save_state("pilot-final"),
        state,
        operation_id=f"{run}:checkpoint:final",
        phase="save_state",
        timeout_seconds=c["timeout_seconds"],
    ).path
    state.assert_dispatchable()
    sampler_path = wait_for_training_phase(
        client.save_weights_for_sampler("pilot-final"),
        state,
        operation_id=f"{run}:checkpoint:final",
        phase="save_sampler",
        timeout_seconds=c["timeout_seconds"],
    ).path
    cp = CheckpointRef(
        BASE_MODEL,
        service.training_session_id,
        client.run_id,
        sampler_path,
        training_path,
        len(batches),
        TOKENIZER_REVISION,
        data.get("manifest_hash", ""),
    )
    state.record({"type": "checkpoint_saved", **cp.to_dict()})
    return cp


def sample_checkpoint(checkpoint: CheckpointRef, protocol: dict) -> dict:
    if (
        checkpoint.base_model != BASE_MODEL
        or checkpoint.tokenizer_revision != TOKENIZER_REVISION
    ):
        raise TrainingNotReady("Checkpoint base/tokenizer mismatch")
    return {
        "name": "B",
        "provider": "fireworks_checkpoint",
        "model": checkpoint.sampler_path,
        "base_model": checkpoint.base_model,
        "session_id": checkpoint.session_id,
        "protocol": dict(protocol),
        "tokenizer_revision": checkpoint.tokenizer_revision,
        "maximum_cost_multiplier": SDK_MAX_SAMPLE_ATTEMPTS,
        "usage_accounts_for_all_attempts": False,
    }


def finalize_session(
    checkpoint: CheckpointRef,
    *,
    retain: bool,
    state: Any,
    service: Any,
    report_exported: bool = False,
    retention_approved: bool = False,
) -> None:
    if not report_exported or not retention_approved:
        raise TrainingNotReady(
            "Exported report and explicit checkpoint retention decision required"
        )
    if service.training_session_id != checkpoint.session_id:
        raise TrainingNotReady("Cleanup service/session mismatch")
    state.record(
        {
            "type": "session_finalize_intent",
            "session_id": checkpoint.session_id,
            "retain": retain,
        }
    )
    if not retain:
        service.delete_training_session()
    state.record(
        {
            "type": "session_finalized",
            "session_id": checkpoint.session_id,
            "retained": retain,
        }
    )


def open_session(
    config: dict, data: dict, state: Any, *, service_factory: Any = None
) -> dict:
    """Create exactly one guarded session/trainer, before base evaluation.

    This is a mutation and must only be called after resolved plan approval.
    Its zero-dollar reservation reflects documented no provisioning/idle fee;
    sampling/training reserve their token costs before each dispatch.
    """
    c, _, _ = _validate(config, data)
    state.assert_dispatchable()
    operation = str(config.get("run_id", "pilot")) + ":session:create"
    state.reserve(operation, Decimal(0))
    state.record(
        {
            "type": "training_session_intent",
            "operation_id": operation,
            "base_model": BASE_MODEL,
        }
    )
    if service_factory is None:
        from fireworks.training.sdk import FiretitanServiceClient

        service_factory = FiretitanServiceClient
    service = service_factory(
        base_url="https://api.fireworks.ai/training/v1/serverless",
        max_retries=0,
        timeout=c["timeout_seconds"],
    )
    state.record(
        {
            "type": "training_session_allocated",
            "operation_id": operation,
            "session_id": service.training_session_id,
        }
    )
    try:
        client = service.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=c["lora_rank"],
            alpha=c["alpha"],
            seed=c["seed"],
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"fireworkshack_run_id": str(config.get("run_id", "pilot"))},
        )
    except Exception as error:
        state.record(
            {
                "type": "trainer_creation_failed",
                "operation_id": operation,
                "session_id": service.training_session_id,
                "exception_type": type(error).__name__,
            }
        )
        try:
            service.delete_training_session()
            state.record(
                {
                    "type": "partial_session_deleted",
                    "session_id": service.training_session_id,
                }
            )
        except Exception as cleanup_error:
            state.record(
                {
                    "type": "partial_session_cleanup_unverified",
                    "session_id": service.training_session_id,
                    "exception_type": type(cleanup_error).__name__,
                }
            )
        raise
    state.record(
        {
            "type": "training_session_created",
            "operation_id": operation,
            "session_id": service.training_session_id,
            "training_run_id": client.run_id,
        }
    )
    state.settle(operation, Decimal(0))
    return {"service": service, "client": client}


class _SDKSampler:
    """OpenAI-shaped callback for providers.generate; caller owns reservation."""

    def __init__(
        self,
        service: Any,
        renderer: NativeRenderer,
        *,
        checkpoint: CheckpointRef | None = None,
    ):
        from fireworks.training.sdk.sampling import DeploymentSampler

        if DeploymentSampler._RETRY_MAX_ATTEMPTS != SDK_MAX_SAMPLE_ATTEMPTS:
            raise TrainingNotReady(
                "SDK sampler attempt ceiling changed; reprice before dispatch"
            )
        self.service, self.renderer, self.checkpoint = service, renderer, checkpoint
        if (
            checkpoint is not None
            and service.training_session_id != checkpoint.session_id
        ):
            raise TrainingNotReady("Sampler service/session mismatch")
        # Native token ids are passed directly; this adapter also satisfies
        # SDK tokenizer decode/encode conventions without transformers/torch.
        native = renderer

        class TokenizerAdapter:
            def encode(self, text, add_special_tokens=False):
                return native._tokens(text)

            def decode(self, tokens, skip_special_tokens=False):
                return native.tokenizer.decode(
                    tokens, skip_special_tokens=skip_special_tokens
                )

        tokenizer = (
            TokenizerAdapter() if hasattr(renderer, "_tokens") else renderer.tokenizer
        )
        kwargs = {"tokenizer": tokenizer}
        if checkpoint is None:
            kwargs["base_model"] = BASE_MODEL
        else:
            kwargs["model_path"] = checkpoint.sampler_path
        self.sampler = service.create_sampling_client(**kwargs)

    def __call__(self, body: dict) -> dict:
        import time

        from fireworks.training import sdk

        reasoning = body.get("reasoning_effort")
        if reasoning not in {"off", "low", "high", "max"}:
            raise TrainingNotReady("Explicit common reasoning mode required")
        tokens = self.renderer.render_prompt(body["messages"], reasoning=reasoning)
        max_tokens = body.get("max_tokens")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise TrainingNotReady("Bounded output tokens required")
        params = sdk.SamplingParams(
            max_tokens=max_tokens, temperature=body["temperature"], top_p=body["top_p"]
        )
        started = time.monotonic()
        result = self.sampler.sample(
            prompt=sdk.ModelInput.from_ints(tokens),
            num_samples=1,
            sampling_params=params,
        ).result(timeout=120)
        if len(result.sequences) != 1:
            raise TrainingNotReady("Expected one sampled sequence")
        sequence = result.sequences[0]
        output = list(sequence.tokens)
        text = self.renderer.decode(output)
        eos = self.renderer.encoder.eos_token
        text = text.split(eos, 1)[0]
        stop = getattr(sequence, "stop_reason", None)
        finish = {"stop": "stop", "length": "length", "max_tokens": "length"}.get(stop)
        final, thinking = text, ""
        if reasoning != "off":
            if "</think>" in text:
                thinking, final = text.split("</think>", 1)
            else:
                thinking, final = text, ""
        return {
            "model": self.checkpoint.sampler_path if self.checkpoint else BASE_MODEL,
            "choices": [
                {
                    "message": {"content": final, "reasoning_content": thinking},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": len(tokens), "completion_tokens": len(output)},
            "backend": "fireworks-training-session",
            "finish_reason_source": "fireworks_sdk_normalized",
            "maximum_cost_multiplier": SDK_MAX_SAMPLE_ATTEMPTS,
            "usage_accounts_for_all_attempts": False,
            "cost_usd": None,
            "usage_source": "submitted_and_returned_token_counts",
            "latency_seconds": time.monotonic() - started,
        }

    def close(self) -> None:
        self.sampler.close()


def make_base_sampler(service: Any, renderer: NativeRenderer) -> Any:
    """Base-only route on an already approved session, no adapter snapshot."""
    return _SDKSampler(service, renderer)


def make_checkpoint_sampler(
    checkpoint: CheckpointRef, service: Any, renderer: NativeRenderer
) -> Any:
    """Sampler bound to the exact saved adapter snapshot."""
    return _SDKSampler(service, renderer, checkpoint=checkpoint)
