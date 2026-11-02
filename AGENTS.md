# Project instructions

Read README.md, docs/experiment-design.md, the implementation plan and HANDOFF.md before execution. Respect any host-level instructions and the existing canonical task record supplied in the private handoff; do not create a duplicate issue.

- This repository is public. Never commit credentials, personal records, chat transcripts, raw provider responses, private run state, datasets or weights.
- Use Python 3.11 or later with uv. Activate `.venv` before Python commands; use `uv add` for dependencies. Pin resolved dependencies in `uv.lock` when implementation begins.
- Work on a `codex/` branch. Check and preserve existing work. Commit only task files, run relevant checks, and push coherent changes before claiming completion.
- The current authorization covers repository setup, planning and handoff. Training, paid inference, uploads, deployment and credit purchases require the resolved run plan and appropriate user approval. Existing credits do not imply an unlimited budget.
- Prefer the supported Fireworks cookbook path. Revalidate SDK interfaces, model availability, tokenizer/renderer, prices and quota before a run. Do not implement a generic remote PyTorch worker or claim layer-hook access.
- Keep training, development and final evaluation separate. Never tune on the final test set. Do not use Dealign evaluation examples as training data.
- Record refusals, harmful compliance, answer completeness, capability, latency and spend as distinct metrics. Higher harmful compliance is not a general quality score.
- Separate author-reported numbers from newly measured results. Label deviations in scoring, checkpoint revision, precision, routing or reasoning settings.
- Return concise findings with an actual report chart or useful architecture diagram, artifact links and verification status. Do not fabricate a successful run.
