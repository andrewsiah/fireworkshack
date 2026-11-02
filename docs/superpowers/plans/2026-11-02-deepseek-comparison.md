# DeepSeek Hosted Comparison Implementation Plan

**Execution checkpoint:** Implementation and the authorized pilot are complete. See [measured report](../../../reports/2026-11-02-pilot/README.md) for actual results, recovery deviations and cleanup. The unchecked steps below preserve the original proposed plan; they are not a current pending execution queue. Actual C used Refuseless with unverified exact Dealign identity; successful B used batch 64 / one optimizer update. No additional paid execution is scheduled.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is the proposed native execution path; obtain user review before implementation.

**Goal:** Build a bounded VM-controlled experiment comparing original DeepSeek 1017, a Fireworks-trained LoRA, and the hosted Dealign model under one evaluation protocol.

**Architecture:** A Python CLI separates offline data preparation from paid execution. Fireworks supplies base/checkpoint sampling and serverless LoRA training; Experiential supplies Dealign inference. An append-only local run record preserves manifests, reservations, responses and aggregate reports for reproducible continuation.

**Tech Stack:** Python 3.11+, uv, Fireworks training SDK/cookbook, OpenAI-compatible HTTP client, pytest, pinned dataset tooling, Matplotlib for exported report figures. Resolve and lock compatible versions during setup.

**Spec:** [Experiment design](../../experiment-design.md).

## Global Constraints

- Exact Fireworks base: `accounts/fireworks/models/deepseek-v4p1-flash-1017`.
- Exact Experiential model: `deepseek-v4.1-flash-uncensored`; base URL `https://api.experientiallabs.ai/v1`.
- Python 3.11 or later; uv environment; activate `.venv` before Python commands.
- Pilot data: 64 reviewed benign-response training rows, 16 separate development rows, maximum rendered length 1,024 tokens; seed 42.
- Pilot evaluations: 16 HarmBench, 57 MMLU, 16 XSTest items; freeze IDs before outputs. For HarmBench use 6 standard, 5 contextual and 5 copyright items. For XSTest use 8 safe and 8 unsafe contrasts. Full final confirmation must remain untouched by development decisions.
- Draft SFT settings for costed approval: rank 8, one epoch, learning rate 1e-4, at most 65,536 training tokens. Validate these settings and the compatible renderer before launch.
- Deadline target: 60 minutes after prerequisites. No paid execution without a separately approved resolved configuration and total USD ceiling.
- Runtime paths: ignored `data/`, `runs/`, `checkpoints/`, `outputs/`, `.local/`; no secrets or raw generations in public Git.
- Use an initial `codex/` branch, run appropriate checks, commit and push coherent changes; do not merge or deploy as an implicit part of implementation.

## Review Focus

1. Missing or misleading credentials/capacity: fail preflight before creating a paid run; never print key values.
2. Unbounded spend/retries: reserve maximum batch cost before dispatch, enforce deadline and finite attempts; delayed/unknown usage retains its reservation.
3. Provider defaults and reasoning-only output: explicit supported settings, pinned Experiential route, separate final text/reasoning/truncation; absent final text is not completed compliance.
4. Evaluation contamination or scoring mismatch: enforce disjoint IDs/content hashes, preserve contextual inputs, and reject mixed logit/generated-answer MMLU protocols.
5. Partial failure or resume: reuse durable item/step IDs, retain pending state after uncertain requests, avoid double training and mark incomplete reports honestly.

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Reproducible dependencies and CLI entry point |
| `configs/pilot.yaml` | Nonsecret proposed configuration; paid execution disabled |
| `src/fireworkshack/config.py` | Typed resolved settings and validation |
| `src/fireworkshack/state.py` | Append-only run events, reservations and resume reconciliation |
| `src/fireworkshack/data.py` | Licensed-source filtering, splitting, overlap checks and manifests |
| `src/fireworkshack/providers.py` | Normalized provider generation and capability checks |
| `src/fireworkshack/evaluate.py` | Common suite formatting and scoring |
| `src/fireworkshack/train.py` | Supported Fireworks SFT recipe adapter and checkpoint lifecycle |
| `src/fireworkshack/report.py` | Aggregate results, uncertainty and standalone charts |
| `src/fireworkshack/cli.py` | Offline defaults and explicit paid-run commands |
| `tests/` | Synthetic fixtures and mocked providers only |

Do not create empty scaffolding for all modules first. Add files with the task that gives them working behavior.

### Task 1: Offline configuration and bounded execution state

**Files:** Create `pyproject.toml`, `configs/pilot.yaml`, `src/fireworkshack/{config,state,cli}.py`, `tests/test_preflight.py`, `tests/test_state.py`; generate `uv.lock`.

**Interfaces:**
- `load_config(path: Path) -> RunConfig`, including model IDs, dataset revision, tracks, per-track decoding/token limits, total USD ceiling, request ceiling, deadline, finite retry counts and checkpoint retention choice.
- `preflight(config: RunConfig, env: Mapping[str, str]) -> PreflightReport`; report capability checks as verified, unverified or failed, never infer training availability from model listing alone.
- `RunState.reserve(operation_id: str, maximum_usd: Decimal) -> None`; raises `BudgetExceeded` before dispatch. `RunState.record(event: RunEvent) -> None` appends and flushes. Uncertain operations remain pending until reconciled.
- CLI: `fireworkshack preflight --config configs/pilot.yaml --offline` and `fireworkshack run --config ... --dry-run`; both default to zero network calls.

- [ ] Write failing tests with a fake transport: offline preflight makes zero requests; missing required key reports its variable name without value; no approved ceiling rejects paid execution; reserving 0.02 against 0.01 remaining raises `BudgetExceeded` with zero dispatches; resumed pending operation cannot dispatch again without reconciliation; expired deadline rejects new work.
- [ ] Run `pytest tests/test_preflight.py tests/test_state.py -q`; observe expected missing-interface failures.
- [ ] Implement minimal typed configuration and state. Use decimal currency, finite retries, atomic manifest writes and append-only run events. Keep approval and secret values out of public configuration.
- [ ] Run those tests and `fireworkshack preflight --config configs/pilot.yaml --offline`; expect tests pass, no provider calls, explicit unverified remote checks.
- [ ] Commit and push the working offline path.

### Task 2: Reviewed training and locked evaluation manifests

**Files:** Create `src/fireworkshack/data.py`, `tests/test_data.py`; extend `cli.py` with `prepare-data` and `freeze-eval` commands.

**Interfaces:**
- `select_training(rows: Iterable[SourceRow], *, revision: str, seed: int = 42) -> DatasetManifest`; produces 64 train and 16 development row IDs and hashes after review/filters.
- `freeze_evaluation(suites: Mapping[str, Sequence[EvalItem]], *, seed: int = 42) -> EvalManifest`; produces pilot IDs in the specified strata and separately reserved final-confirmation IDs.
- `assert_disjoint(train: DatasetManifest, evaluation: EvalManifest) -> None` checks source IDs and normalized content overlap, with a documented near-duplicate review step.
- `EvalItem` contains `item_id`, `suite`, `prompt`, `context`, `expected_answer`, `category`, `split`; runtime content is private. Manifests include source revisions and selection parameters.

- [ ] Write failing synthetic tests: harmful prompt, harmful response, refusal response and missing response are excluded; 79 valid rows fails the 80-row requirement; fixed seed yields the same IDs; shared/paraphrased examples are flagged for removal/review; contextual input survives serialization; HarmBench selection has counts 6/5/5; XSTest has 8/8.
- [ ] Run `pytest tests/test_data.py -q`; observe expected failures.
- [ ] Implement filtering only after inspecting the actual licensed schema and label values. Include a review manifest that must be completed before data becomes training-ready. Fail when access/terms or review is missing, rather than auto-accepting conditions.
- [ ] Run tests; run offline preparation against synthetic fixtures and check manifest hashes and zero paid calls. Real data preparation waits for access and review.
- [ ] Commit and push code plus synthetic fixtures; do not commit real dataset rows.

### Task 3: Common provider adapters and evaluation

**Files:** Create `src/fireworkshack/providers.py`, `src/fireworkshack/evaluate.py`, `tests/test_providers.py`, `tests/test_evaluate.py`; extend `cli.py` with `evaluate`.

**Interfaces:**
- `generate(arm: Arm, item: EvalItem, protocol: EvalProtocol, state: RunState) -> GenerationResult`.
- `GenerationResult` records stable request/item ID, model, backend/route, protocol hash, final text, reasoning separately when supplied, finish reason, token usage, actual/reserved cost, latency and error state; store privately.
- `score(item: EvalItem, result: GenerationResult, judge: JudgeConfig) -> ItemScore`; returns distinct capability, refusal, harmful-compliance, completeness and truncation fields with judge version.
- `evaluate(arm: Arm, manifest: EvalManifest, protocol: EvalProtocol, state: RunState) -> EvaluationSummary` resumes known completed items and reports pending/failed items separately.

- [ ] Write failing mocked tests: model IDs and protocol fields match configuration; Experiential route is explicit and fallback disabled; unsupported reasoning/logprob settings fail rather than disappear; empty final answer with reasoning is incomplete; truncated answer remains flagged; contextual HarmBench prompt is formatted correctly; mixed MMLU scoring modes are rejected; missing/error items remain in denominators with separate coverage.
- [ ] Run `pytest tests/test_providers.py tests/test_evaluate.py -q`; observe expected failures.
- [ ] Implement against current primary SDK/API docs. Normalize outputs but preserve relevant serving differences. Do not count every nonrefusal as successful harmful compliance; wire official evaluation/independent judge semantics separately from Dealign's published rubric.
- [ ] Run tests and a dry evaluation using mocked A/B/C outputs. Expect identical item IDs/protocol hashes and explicit incomplete/error fields.
- [ ] Commit and push. Live smoke calls wait for the complete paid-run plan and approval.

### Task 4: Fireworks SFT and checkpoint sampling

**Files:** Create `src/fireworkshack/train.py`, `tests/test_train.py`; extend `cli.py` with `train` and resume controls.

**Interfaces:**
- `train_sft(config: RunConfig, data: DatasetManifest, state: RunState) -> CheckpointRef`.
- `CheckpointRef` privately stores session/run identity, base model, step, checkpoint identity, compatible renderer/tokenizer revision and config/data hashes.
- `sample_checkpoint(checkpoint: CheckpointRef, protocol: EvalProtocol) -> Arm` exposes B through the same evaluator as Task 3.
- `finalize_session(checkpoint: CheckpointRef, *, retain: bool, state: RunState) -> None` executes only the approved checkpoint retention/cleanup policy.

Starting reference: `training/examples/sft/train_sft_serverless.py` in fw-ai/cookbook, inspected research revision `57968a2237b663be33078d2747a28463983ae134`. Inspect its current dependencies and exact SDK interfaces before adapting; do not copy unrelated example model defaults.

- [ ] Write failing mocked tests: wrong base rejected; sequence >1,024 tokens or total >65,536 training tokens rejected; optimizer update without a durable step intent rejected; unknown pending update on resume stops for reconciliation; B receives the saved checkpoint identity; deletion cannot run before approved retention decision/report export.
- [ ] Run `pytest tests/test_train.py -q`; observe expected failures.
- [ ] Implement the minimal supported serverless SFT path. Verify renderer, assistant-only loss/token masks, optimizer values and session capabilities before paid work. Record raw/effective training settings. A model metadata flag is not a successful training-session test.
- [ ] Run mocked tests and `fireworkshack train --config configs/pilot.yaml --dry-run`; expect planned token/request counts and zero API writes.
- [ ] Commit and push. Do not create a training client during dry-run.

### Task 5: Reviewable report and controlled pilot execution

**Files:** Create `src/fireworkshack/report.py`, `tests/test_report.py`; extend `cli.py` with `report`; update `README.md` and `HANDOFF.md` with verified usage.

**Interfaces:**
- `build_report(state: RunState, output_dir: Path) -> ReportArtifacts`, producing Markdown, aggregate CSV and standalone PNG charts. Output stays ignored until reviewed for publication.
- The comparison chart displays A/B/C capability and refusal/compliance metrics separately, sample sizes and intervals; runtime/cost get separate panels. Published Dealign claims, if displayed, are labeled external author-reported references.

- [ ] Write failing synthetic tests: absent B is “not run,” never zero accuracy; reasoning-only results do not count as completed answers; per-suite denominators and coverage are shown; published claims cannot mix with measured scores; costs preserve reservations and mark unsettled usage; all-error/one-item samples still produce an honest report.
- [ ] Run `pytest tests/test_report.py -q`; observe expected failures.
- [ ] Implement aggregate reporting with Wilson intervals for proportions and paired-item capability deltas. Do not imply statistical superiority from the pilot. Export a chart that can be opened without a running app.
- [ ] Run `pytest -q`, offline CLI dry-runs and synthetic report generation; inspect the actual chart. Check no credentials, dataset bodies or raw outputs enter tracked files. Commit and push.
- [ ] Present the resolved configuration, current prices/worst-case total cost, both-provider credential readiness, renderer/route checks, retry/request/token caps, judge cost, deadline, output storage and checkpoint retention policy. Obtain approval required by the training workflow before live calls.
- [ ] After approval, run the predeclared one-hour pilot. Save durable events and actual usage. Stop at limits; no automatic top-up or model/dataset substitution.
- [ ] Generate and inspect the actual report and figure. Record what was measured, unresolved precision/backend differences and incomplete steps. Publish only reviewed compatible aggregates on a pushed task branch; do not merge or deploy implicitly.

## Plan review and handoff gate

This plan is ready for user review. Repository publication and VM handoff do not approve implementation or paid execution. Suggested implementation method is native task-by-task execution because the small harness shares configuration/state interfaces throughout. Do not automatically spawn agents or duplicate a live experiment.

Before launch, resolve the remaining decisions explicitly: dataset access and approval of the benign-response pilot's limited objective; exact supported per-track reasoning/output limits; MMLU scoring support; total USD ceiling; checkpoint retention. If the user wants a broader Dealign-like training objective immediately, revise the data and evaluation design before running instead of relabeling the safe-response pilot.
