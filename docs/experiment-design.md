# DeepSeek comparison design

Research checkpoint: November 2, 2026. Provider behavior and pricing must be revalidated before execution.

## Question and scope

Compare the same upstream DeepSeek model under three conditions: original weights, a Fireworks-trained LoRA, and Dealign's released weight modification. Measure behavior and capability directly through hosted endpoints. The intended broader question is whether alternative training can approach Dealign's reported behavior with less capability loss. The first pilot establishes connectivity, data handling, training and evaluation; it cannot answer that broader question statistically.

This is an alternative-method comparison. Dealign describes direct weight-level edits with no LoRA, SFT, DPO or RL. The inspected release contains inference code but no published intervention script or calibration set. Do not call this an exact method replication or claim to use “Dealign's training dataset.”

## Architecture

| Arm | Model | Execution |
|---|---|---|
| A: original | `accounts/fireworks/models/deepseek-v4p1-flash-1017` | Fireworks base sampling |
| B: ours | Same upstream model plus our LoRA checkpoint | Fireworks Serverless Training and in-session checkpoint sampling |
| C: Dealign | `deepseek-v4.1-flash-uncensored` | Experiential OpenAI-compatible inference |

The CPU VM prepares batches, runs the controller, calls provider APIs and scores responses. Fireworks performs remote forward/backward/optimizer work. Use the training session sampler for B rather than provisioning a permanent on-demand deployment for a pilot. Export/promote a checkpoint only if retention is explicitly part of the approved plan; deleting a session removes access to its checkpoints.

Experiential base URL: `https://api.experientiallabs.ai/v1`; authentication variable: `EXPLABS_API_KEY`. Its model listing links the exact Dealign FP8 release. Thinking is enabled by default. Its gateway can choose between provider routes: select a stable route through an authenticated route handle, disable fallbacks for the comparison and record the observed backend. Listing verification does not establish live access or an immutable checkpoint revision.

Fireworks serverless supports LoRA training, including SFT, preference optimization, RL and custom loops. It does not expose arbitrary model-weight surgery or general-purpose GPU shell access. Custom chat-model losses can use the documented token-logprob interface; do not assume arbitrary layer activations are accessible. Dedicated/full-parameter training and Heretic are outside the first pilot.

## Dataset decision

**Proposed first dataset: `allenai/wildguardmix`, configuration `wildguardtrain`, benign-response subset.** Select rows with a response, a benign prompt label, a nonharmful response label and a nonrefusal response label. Verify exact column names and label encodings from the pinned dataset revision before filtering. Audit examples manually; labels are imperfect.

Create 64 training examples and 16 separate development examples, each rendered to at most 1,024 tokens. Record retained/rejected row IDs, source category, revision, content hash and filtering reasons. Split before experimentation, deduplicate exact and near-duplicate prompts, and remove overlap with every evaluation suite. If 80 valid examples cannot be selected, stop and report the limitation rather than relaxing the filters silently.

Why this first: it supplies existing prompt-response supervision without needing a teacher-generation stage. It tests helpful responses on benign inputs. It is not expected to reproduce all behavior claimed for an unrestricted weight edit; avoid interpreting a negative result as proof that Fireworks cannot support a different training objective.

WildGuardMix requires accepting its access conditions. Do not download or redistribute it until access and permitted use are resolved. Keep full data local and ignored; publish only compatible provenance and selection logic. Dataset choice is a proposed pilot, not a confirmed user decision about the broader study.

A later DPO experiment is reasonable only if suitable, independently reviewed chosen/rejected pairs are available. An RL experiment requires a defensible reward and additional sampling budget. Neither is needed to validate the first workflow. Do not train against the final benchmark or optimize on its judge score.

## Evaluation protocol

Use one adapter-neutral harness with identical item IDs and judge version for A, B and C. Freeze system message, user/context formatting, reasoning mode, temperature, top-p and output limit before requests. Do not rely on endpoint defaults. Record the effective supported settings; a rejected or silently unsupported setting is a preflight failure or a documented protocol difference.

### Full comparison after the pilot

| Suite | Measurement | Reproduction limit |
|---|---|---|
| HarmBench text test, 320 items | Refusal labels and official harmful-compliance audit, by functional category | Dealign's COMPLY rubric differs from standard classifier ASR |
| MMLU, 57 subjects x 20 questions | Capability accuracy | Exact Dealign row IDs/selection seed not published in inspected release |
| XSTest, entire suite | Benign overrefusal and unsafe contrasts, separately | Additional evaluation, not a Dealign-reported score |

The official HarmBench test CSV at revision `8e1604d1171fe8a48d8febecd22f600e462bdcdd` contains 159 standard, 81 contextual and 80 copyright items. Preserve `BehaviorID` and contextual input. Never put final test items in training, development or early stopping.

Dealign reports MMLU 87.63% for the original and 83.77% for its modification, a 3.86 percentage-point decrease. Its 1,140-question protocol uses no reasoning and first-token letter-logit scoring. Verify logprob availability on both providers. If equivalent scoring is unavailable, use the same generated-letter accuracy method across all three arms and label it a new protocol; do not directly equate it to the published logit score. Select and publish our own fixed question IDs before running if the original selection cannot be obtained.

Dealign reports 320/320 COMPLY on HarmBench with maximum reasoning, greedy decoding and a 200-token generation limit. Its rubric can grade early reasoning intent when generation ends before the reasoning terminator. Report refusal, completed-answer compliance, reasoning-only intent and truncation separately. Author-reported 100% COMPLY is not independently verified here and is not automatically the official HarmBench metric.

Evaluate safety and benign helpfulness separately. No single score should make increased harmful compliance appear equivalent to higher model quality. A defensible result shows the tradeoff against capability, completeness, runtime and cost. With 100% author-reported compliance there is no numerical headroom on that particular reported measure; the real comparison requires common scoring and retained capability.

### Pilot subset

Before any model output is inspected, freeze disjoint pilot subsets: 16 HarmBench items stratified across its three functional categories, 57 MMLU questions (one per subject), and 16 XSTest items balanced across safe/unsafe labels. Use a fixed seed of 42 and save exact item IDs. These subsets are development evidence only; the later final confirmation excludes every item used to make training decisions or uses a separately predeclared untouched set. Never present 89 examples as a full benchmark.

For the pilot capability track use reasoning disabled and a short output cap. For behavior tracks predeclare a supported common reasoning setting and cap; maximum reasoning with a short cap may produce no final answer. Report those outcomes rather than counting them as successful completions. A 200-token Dealign-protocol audit may be reported separately from a longer completed-answer protocol. Choose exact per-track limits during costed configuration.

## One-hour target and budget policy

The one-hour clock starts only after credentials, dataset access, renderer and quota are ready. This is a proposed schedule, not a completed or guaranteed run:

| Minutes | Deliverable |
|---|---|
| 0-10 | Preflight, freeze manifests/settings, minimal endpoint smoke tests |
| 10-20 | Measure A and C on locked pilot subsets |
| 20-40 | One bounded SFT adapter pass on 64 examples |
| 40-55 | Measure B with the identical harness |
| 55-60 | Export aggregate report, actual costs, limitations and checkpoint decision |

Draft training defaults for configuration review: LoRA rank 8, one epoch, learning rate 1e-4, maximum rendered sequence length 1,024, maximum 65,536 training tokens. These are starting hypotheses, not validated optimal hyperparameters or an approved run. Use supported optimizer/microbatch defaults from the pinned recipe and record their resolved values.

No budget has been approved. Existing provider credits are not a spending cap. Before any paid call, resolve a USD ceiling across both providers, per-track request/token limits, judge costs, allowed retries, deadline, checkpoint policy and clean-up. Estimate the worst-case reserve for each batch using current prices; refuse dispatch if it would exceed the remaining approved budget. Reconcile actual usage without assuming delayed billing is instantly settled. Do not enable automatic top-ups or start a paid deployment.

Stop on access failure, unsupported renderer/protocol, insufficient capacity, exhausted budget or deadline. Save partial progress and label it incomplete. Do not silently substitute a different model, training dataset or scoring method.

## Artifacts and acceptance

Public: code, locked dependency versions, provider/config schemas with placeholders, compatible dataset manifests, frozen evaluation IDs, aggregate reports and charts after review. Private/ignored: credentials, dataset bodies, raw prompts/responses, reasoning, checkpoint handles and billing/account details.

The report must show sample counts/denominators, per-arm settings and backend, uncertainty intervals for measured proportions, paired capability differences, incomplete/truncated/error counts, inference and training cost separately, and elapsed time. Backend/precision differences remain a limitation even with matching prompts. A small pilot supports a workflow claim only.

## Sources and verification record

- [Fireworks Serverless Training](https://docs.fireworks.ai/fine-tuning/training-api/serverless): exact model listed, LoRA and custom loops, in-session sampling and session cleanup.
- [Fireworks Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated): different access and full-parameter surface; not used by the pilot.
- [Fireworks model catalog](https://docs.fireworks.ai/fine-tuning/fine-tuning-models): revalidate model eligibility. Prior authenticated model metadata check returned HTTP 200 with LoRA eligibility flags; actual training capacity is untested.
- [Dealign model release](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8), inspected revision `c19ca5ca4eaeb79d29fdf08aea8327da95b1116a`: method and author-reported benchmarks.
- [Experiential model listing](https://platform.experientiallabs.ai/models/deepseek-v4.1-flash-uncensored): checkpoint attribution and model slug.
- [Experiential routing](https://platform.experientiallabs.ai/docs/waterfall): route pinning and fallback controls.
- [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix): access-conditioned labeled prompt/response corpus.
- [HarmBench](https://github.com/centerforaisafety/HarmBench): official data and evaluation code.
- [MMLU](https://huggingface.co/datasets/cais/mmlu): capability data.
- [XSTest](https://github.com/paul-rottger/xstest): exaggerated safety behavior evaluation.
- [Fireworks cookbook](https://github.com/fw-ai/cookbook/tree/57968a2237b663be33078d2747a28463983ae134/training): inspected starting revision; revalidate compatibility before installation.

The Fireworks training skills were installed on the research host. Their presence, SDK dependencies and credentials on a different execution host are not established by this repository. No model inference, training, dataset download or checkpoint promotion has been performed for this experiment.
