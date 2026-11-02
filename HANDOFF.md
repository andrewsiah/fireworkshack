# Continuation checklist

This repository currently contains planning documents only. Read the design and implementation plan before adding code.

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
