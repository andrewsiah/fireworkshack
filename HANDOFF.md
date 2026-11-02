# Continuation checklist

This repository now contains a bounded Python controller, mocked checks and a read-only local HTML dashboard. Read the design, implementation plan, current canonical task journal and any existing private run state before execution. A successful offline check is not a model-performance result.

## Implementation checkpoint

- Use `uv sync --locked` in the activated `.venv`; the lightweight `fireworks-ai[training-sdk]` extra avoids local GPU dependencies.
- `fireworkshack preflight --offline` and `run --dry-run` make zero provider calls. Public config keeps paid execution disabled.
- Dataset access and review are explicit; private manifests bind exact revisions, IDs and hashes. `scripts/prepare_private_data.py` requires existing gated access and content-bound review records. It freezes pilot IDs and reserves separate final confirmation.
- Live entry is `run --execute` with exact private config/approval hashes. The exclusive controller lease and existing controller record prevent duplicate restarts. Unknown requests/optimizer updates require reconciliation.
- A and B use the same training session, with base-only sampling before updates. Seven possible internal SDK sampling attempts are fully reserved; total billed usage remains unsettled until reconciled. No deployment is provisioned.
- Reports preserve missing/unjudged coverage. Agent audit criteria are in `docs/scoring-protocol.md`; author-reported Dealign scores are not newly measured results.
- Local dashboard: `python -m fireworkshack.dashboard --port 8765`. Serve only the fixed assets and sanitized aggregate API. Private data, raw outputs and credential files must never become static assets.
- C currently uses the explicitly supplied Refuseless endpoint, with exact Dealign provenance unverified. Never silently fall back to another provider or claim an immutable revision from the catalog name.

## First actions

1. Reconcile the canonical task and current branch with the private handoff. Confirm no other worker owns the same files or live runs.
2. Check the VM instructions, Python/uv, GitHub authentication, and presence of runtime credentials without printing their values. Nothing here proves that credentials were provisioned on this host.
3. Review the plan with the user, including the limited scope of the safe-response pilot. Implementation execution method and paid-run approval are separate from publishing these documents.
4. Resolve WildGuardMix access and dataset suitability. Do not accept access terms on the user's behalf or silently replace the research objective when data is unavailable.
5. Implement offline configuration, dataset selection and the common evaluator before enabling paid calls. Use mocked provider tests and a default dry-run mode.
6. Present the exact model, data manifest, settings, cost bound, token/request limits, one-hour deadline and cleanup policy before the first paid run.

## Known facts and open checks

- Fireworks publicly lists the exact upstream DeepSeek V4.1 Flash 1017 for serverless LoRA. A prior read-only model API check also returned LoRA eligibility flags. A training session has not been created.
- Experiential's public listing explicitly identifies Dealign's FP8 model and links the release. Live authenticated inference, checkpoint revision and MMLU logprob support have not been checked.
- Dealign's release describes weight surgery without SFT/DPO/RL. No edit script or calibration dataset was found in the inspected public release. This project compares outcomes from a different method.
- The first 64-row benign-response SFT pilot does not test the full broad-compliance hypothesis. A later experiment requires a separately justified, licensed training dataset and locked evaluation protocol.
- The small pilot is a one-hour target after prerequisites, not a runtime guarantee. Queue time, access issues or missing compatible renderer can stop the attempt.
- Raw responses and private run state belong in ignored runtime storage. Publish only reviewed, license-compatible aggregate results and manifests without sensitive content.

## Done for the first pilot

A report identifies the exact three arms, dataset revisions and row IDs, a reproducible before/after evaluation, actual cost and runtime, truncations/failures, and the limits of the conclusion. If training cannot start, report a preflight failure rather than a model result.
