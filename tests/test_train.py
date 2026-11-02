from decimal import Decimal
from types import SimpleNamespace

import pytest

from fireworkshack.train import (
    BASE_MODEL,
    TOKENIZER_REVISION,
    CheckpointRef,
    TrainingNotReady,
    finalize_session,
    prepare_training_data,
    sample_checkpoint,
    train_sft,
)


class State:
    def __init__(self):
        self.events = []
        self.reserved = {}
        self.pending = set()

    def assert_dispatchable(self):
        pass

    def reserve(self, operation_id, maximum_usd):
        if operation_id in self.reserved:
            raise RuntimeError("pending operation requires reconciliation")
        self.reserved[operation_id] = maximum_usd

    def record(self, event):
        self.events.append(event)

    def settle(self, operation_id, actual_usd):
        pass


def config():
    return {
        "run_id": "test",
        "training": {
            "base_model": BASE_MODEL,
            "renderer_verified": True,
            "tokenizer_revision": TOKENIZER_REVISION,
            "max_seq_len": 1024,
            "max_train_tokens": 65536,
            "lora_rank": 8,
            "epochs": 1,
            "learning_rate": 0.0001,
            "batch_size": 8,
            "train_usd_per_million": "4.98",
        },
    }


def test_partial_session_creation_is_recorded_and_cleaned():
    from fireworkshack.train import open_session

    service = SimpleNamespace(training_session_id="partial-known", deleted=False)

    def fail(**kwargs):
        raise RuntimeError("trainer unavailable")

    def delete():
        service.deleted = True

    service.create_lora_training_client = fail
    service.delete_training_session = delete
    state = State()
    with pytest.raises(RuntimeError):
        open_session(config(), data(), state, service_factory=lambda **kwargs: service)
    assert service.deleted
    assert any(e.get("session_id") == "partial-known" for e in state.events)


def data():
    return {
        "ready": True,
        "manifest_hash": "abc",
        "review": {"approved": True},
        "train": [
            {"item_id": str(i), "token_ids": [1, 2, 3], "weights": [0, 0, 1]}
            for i in range(64)
        ],
        "dev": [{"item_id": "d" + str(i)} for i in range(16)],
        "renderer_revision": TOKENIZER_REVISION,
    }


def test_dry_run_has_exact_cost_and_never_constructs_client():
    result = train_sft(
        config(),
        data(),
        State(),
        service_factory=lambda **kw: pytest.fail("network"),
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["steps"] == 8
    assert Decimal(result["maximum_training_usd"]) == Decimal("0.00095616")


def test_full_batch_recovery_is_one_update_with_documented_wait():
    c = config()
    c["training"].update(batch_size=64, timeout_seconds=1800)
    plan = train_sft(c, data(), State(), dry_run=True)
    assert plan["rows"] == 64 and plan["steps"] == 1


@pytest.mark.parametrize(
    "setting,value",
    [
        ("base_model", "wrong"),
        ("lora_rank", 16),
        ("epochs", 2),
        ("max_seq_len", 2048),
        ("max_train_tokens", 65537),
    ],
)
def test_unapproved_settings_fail(setting, value):
    c = config()
    c["training"][setting] = value
    with pytest.raises(TrainingNotReady):
        train_sft(c, data(), State())


def test_oversized_or_unrendered_rows_stop_before_client():
    d = data()
    d["train"][0]["token_ids"] = list(range(1025))
    d["train"][0]["weights"] = [1] * 1025
    with pytest.raises(TrainingNotReady):
        train_sft(config(), d, State())
    d = data()
    d["train"][0].pop("weights")
    with pytest.raises(TrainingNotReady):
        train_sft(config(), d, State())


def test_review_renderer_and_readiness_are_required():
    d = data()
    d["ready"] = False
    with pytest.raises(TrainingNotReady):
        train_sft(config(), d, State())
    c = config()
    c["training"]["renderer_verified"] = False
    with pytest.raises(TrainingNotReady):
        train_sft(c, data(), State())


class Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class TrainingClient:
    run_id = "run-test"

    def __init__(self, state):
        self.state = state
        self.steps = 0

    def forward_backward(self, datums, loss):
        assert self.state.events[-1]["type"] == "training_step_intent"
        assert loss == "cross_entropy"
        return Future(SimpleNamespace(metrics={"loss:sum": 1}))

    def optim_step(self, adam, **kw):
        assert kw["emit_grad_norm_metrics"] is True
        self.steps += 1
        return Future(SimpleNamespace(metrics={"grad_norm": 1}))

    def save_state(self, name):
        return Future(SimpleNamespace(path="state/" + name))

    def save_weights_for_sampler(self, name):
        return Future(SimpleNamespace(path="sampler/" + name))


class Service:
    training_session_id = "ts-test"

    def __init__(self, state):
        self.state = state
        self.trainer = TrainingClient(state)
        self.deleted = False

    def create_lora_training_client(self, **kw):
        assert kw["base_model"] == BASE_MODEL
        assert kw["alpha"] == 32
        return self.trainer

    def delete_training_session(self):
        self.deleted = True

    def create_sampling_client(self, **kw):
        self.sample_path = kw["model_path"]
        return SimpleNamespace(close=lambda: None)


def test_live_injected_loop_records_intents_and_returns_exact_snapshot():
    s = State()
    service = Service(s)
    result = train_sft(
        config(), data(), s, dry_run=False, service_factory=lambda **kw: service
    )
    assert isinstance(result, CheckpointRef)
    assert result.session_id == "ts-test" and result.step == 8
    assert result.sampler_path == "sampler/pilot-final"
    assert sum(x["type"] == "training_step_intent" for x in s.events) == 8
    assert not service.deleted


def test_unknown_pending_update_cannot_dispatch_again():
    s = State()
    s.reserved["test:train:0"] = Decimal(1)
    service = Service(s)
    with pytest.raises(RuntimeError, match="reconciliation"):
        train_sft(
            config(), data(), s, dry_run=False, service_factory=lambda **kw: service
        )
    assert service.trainer.steps == 0


def test_cleanup_requires_report_and_explicit_retention_decision():
    s = State()
    service = Service(s)
    cp = CheckpointRef(
        BASE_MODEL,
        "ts-test",
        "run-test",
        "sampler/pilot-final",
        "state/final",
        8,
        TOKENIZER_REVISION,
        "abc",
    )
    with pytest.raises(TrainingNotReady):
        finalize_session(
            cp,
            retain=False,
            state=s,
            service=service,
            report_exported=False,
            retention_approved=True,
        )
    with pytest.raises(TrainingNotReady):
        finalize_session(
            cp,
            retain=False,
            state=s,
            service=service,
            report_exported=True,
            retention_approved=False,
        )
    finalize_session(
        cp,
        retain=False,
        state=s,
        service=service,
        report_exported=True,
        retention_approved=True,
    )
    assert service.deleted


def test_sampling_arm_uses_exact_snapshot():
    cp = CheckpointRef(
        BASE_MODEL,
        "ts-test",
        "run-test",
        "sampler/exact",
        "state/final",
        8,
        TOKENIZER_REVISION,
        "abc",
    )
    arm = sample_checkpoint(cp, {"reasoning": "off", "max_tokens": 128})
    assert arm["model"] == cp.sampler_path
    assert arm["provider"] == "fireworks_checkpoint"
    assert arm["base_model"] == BASE_MODEL


def test_prepare_uses_real_renderer_masks_and_preserves_review():
    class R:
        revision = TOKENIZER_REVISION

        def render_supervised(self, prompt, response):
            return {"token_ids": [1, 2, 3], "weights": [0, 0, 1]}

    rows = [
        {
            "item_id": str(i),
            "prompt": "hello",
            "response": "hi",
            "review_approved": True,
        }
        for i in range(80)
    ]
    manifest = {
        "train": rows[:64],
        "dev": rows[64:],
        "review": {"approved": True},
        "ready": False,
    }
    prepared = prepare_training_data(manifest, R())
    assert prepared["ready"] and prepared["train_tokens"] == 192
    assert prepared["train"][0]["weights"] == [0, 0, 1]


def test_tokenizer_artifacts_belong_to_exact_1017_release():
    from fireworkshack.train import TOKENIZER_REPOSITORY

    assert TOKENIZER_REPOSITORY == "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def test_open_session_is_guarded_and_does_not_train():
    from fireworkshack.train import open_session

    state = State()
    service = Service(state)
    session = open_session(
        config(), data(), state, service_factory=lambda **kw: service
    )
    assert session["service"] is service and session["client"] is service.trainer
    assert service.trainer.steps == 0
    assert state.events[-1]["type"] == "training_session_created"


def test_sampler_callbacks_use_base_route_and_exact_checkpoint():
    from fireworkshack.train import make_base_sampler, make_checkpoint_sampler

    class Renderer:
        tokenizer = None
        encoder = SimpleNamespace(eos_token="<eos>")

        def render_prompt(self, messages, reasoning="off"):
            return [1, 2, 3]

        def decode(self, tokens):
            return "Hello.<eos>"

    calls = []

    class Sample:
        def sample(self, **kwargs):
            return Future(
                SimpleNamespace(
                    sequences=[SimpleNamespace(tokens=[4, 5], stop_reason="stop")]
                )
            )

        def close(self):
            pass

    class S:
        training_session_id = "ts-test"

        def create_sampling_client(self, **kwargs):
            calls.append(kwargs)
            return Sample()

    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "off",
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
    }
    base = make_base_sampler(S(), Renderer())
    result = base(body)
    assert calls[-1]["base_model"] == BASE_MODEL
    assert result["choices"][0]["message"]["content"] == "Hello."
    assert result["usage"]["prompt_tokens"] == 3
    cp = CheckpointRef(
        BASE_MODEL,
        "ts-test",
        "run-test",
        "sampler/exact",
        "state/final",
        8,
        TOKENIZER_REVISION,
        "abc",
    )
    tuned = make_checkpoint_sampler(cp, S(), Renderer())
    tuned(body)
    assert calls[-1]["model_path"] == "sampler/exact"
    base.close()
    tuned.close()


def test_sampler_preserves_reasoning_only_and_unknown_stop():
    from fireworkshack.train import make_base_sampler

    class Renderer:
        tokenizer = None
        encoder = SimpleNamespace(eos_token="<eos>")

        def render_prompt(self, messages, reasoning="off"):
            return [1]

        def decode(self, tokens):
            return "Still thinking"

    class Sample:
        def sample(self, **kwargs):
            return Future(
                SimpleNamespace(
                    sequences=[SimpleNamespace(tokens=[4], stop_reason=None)]
                )
            )

        def close(self):
            pass

    service = SimpleNamespace(create_sampling_client=lambda **kwargs: Sample())
    callback = make_base_sampler(service, Renderer())
    result = callback(
        {
            "messages": [],
            "reasoning_effort": "high",
            "max_tokens": 128,
            "temperature": 0,
            "top_p": 1,
        }
    )
    assert result["choices"][0]["message"]["content"] == ""
    assert result["choices"][0]["message"]["reasoning_content"] == "Still thinking"
    assert result["choices"][0]["finish_reason"] is None


def test_sampler_labels_sdk_normalized_finish_reason():
    from fireworkshack.train import make_base_sampler

    class Renderer:
        tokenizer = None
        encoder = SimpleNamespace(eos_token="<eos>")

        def render_prompt(self, messages, reasoning="off"):
            return [1]

        def decode(self, tokens):
            return "hello"

    sample = SimpleNamespace(
        sample=lambda **kwargs: Future(
            SimpleNamespace(sequences=[SimpleNamespace(tokens=[2], stop_reason="stop")])
        ),
        close=lambda: None,
    )
    result = make_base_sampler(
        SimpleNamespace(create_sampling_client=lambda **kwargs: sample), Renderer()
    )(
        {
            "messages": [],
            "reasoning_effort": "off",
            "max_tokens": 32,
            "temperature": 0,
            "top_p": 1,
        }
    )
    assert result["finish_reason_source"] == "fireworks_sdk_normalized"


def test_prompt_tokens_may_not_receive_training_weight():
    d = data()
    d["train"][0]["weights"] = [1, 1, 1]
    with pytest.raises(TrainingNotReady, match="assistant"):
        train_sft(config(), d, State())


def test_checkpoint_arm_reserves_all_sdk_attempts():
    cp = CheckpointRef(
        BASE_MODEL,
        "ts-test",
        "run-test",
        "sampler/exact",
        "state/final",
        8,
        TOKENIZER_REVISION,
        "abc",
    )
    arm = sample_checkpoint(cp, {"reasoning": "off", "max_tokens": 32})
    assert arm["maximum_cost_multiplier"] == 7
    assert arm["usage_accounts_for_all_attempts"] is False
