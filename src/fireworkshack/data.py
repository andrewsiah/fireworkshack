"""Deterministic private data manifests and conservative overlap checks.

Dataset bodies belong in ignored runtime storage. ``public_manifest`` removes
text; content hashes are provenance, not a license to redistribute source rows.
Lexical near-duplicate screening flags review candidates, not all paraphrases.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from difflib import SequenceMatcher


class DataValidationError(ValueError):
    """Input fails a declared data contract."""


def normalize_prompt(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _near(a: str, b: str) -> bool:
    if a == b:
        return True
    aa, bb = set(a.split()), set(b.split())
    if not aa or not bb:
        return False
    overlap = len(aa & bb) / len(aa | bb)
    if overlap >= 0.8:
        return True
    return (
        overlap >= 0.4
        and min(len(a), len(b)) / max(len(a), len(b)) >= 0.75
        and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.9
    )


def select_training(rows: Iterable[dict], *, revision: str, seed: int = 42) -> dict:
    """Select 64 train/16 development rows after strict labels and deduplication.

    Review and rendered-token evidence must be provided per row to become ready.
    Selection never turns model-generated dataset labels into a human review.
    """
    if not revision:
        raise DataValidationError("Pinned dataset revision is required")
    candidates, rejected = [], []
    seen_ids, seen_prompts = set(), set()
    for index, row in enumerate(rows):
        item_id = str(row.get("item_id", row.get("row_id", f"wildguardtrain:{index}")))
        prompt, response = row.get("prompt"), row.get("response")
        reason = None
        if not isinstance(prompt, str) or not prompt.strip():
            reason = "missing_prompt"
        elif not isinstance(response, str) or not response.strip():
            reason = "missing_response"
        elif row.get("prompt_harm_label") != "unharmful":
            reason = "prompt_not_verified_unharmful"
        elif row.get("response_harm_label") != "unharmful":
            reason = "response_not_verified_unharmful"
        elif row.get("response_refusal_label") != "compliance":
            reason = "response_not_verified_compliance"
        elif item_id in seen_ids:
            reason = "duplicate_id"
        elif normalize_prompt(prompt) in seen_prompts:
            reason = "duplicate_normalized_prompt"
        if reason:
            rejected.append({"item_id": item_id, "reason": reason})
            continue
        seen_ids.add(item_id)
        seen_prompts.add(normalize_prompt(prompt))
        candidates.append(
            {
                "item_id": item_id,
                "prompt": prompt,
                "response": response,
                "context": "",
                "category": str(row.get("subcategory", "unknown")),
                "revision": revision,
                "prompt_hash": _hash(normalize_prompt(prompt)),
                "content_hash": _hash({"prompt": prompt, "response": response}),
                "review_approved": row.get("review_approved") is True,
                "rendered_tokens": row.get("rendered_tokens"),
                "renderer_revision": row.get("renderer_revision"),
            }
        )
    # Hash ordering remains stable when an audited rejection is removed; the
    # replacement comes next without reshuffling already reviewed examples.
    candidates.sort(key=lambda item: _hash({"seed": seed, "item_id": item["item_id"]}))
    selected, selected_normalized = [], []
    for candidate in candidates:
        normalized = normalize_prompt(candidate["prompt"])
        if any(_near(normalized, old) for old in selected_normalized):
            rejected.append(
                {
                    "item_id": candidate["item_id"],
                    "reason": "near_duplicate_review_required",
                }
            )
            continue
        selected.append(candidate)
        selected_normalized.append(normalized)
        if len(selected) == 80:
            break
    if len(selected) < 80:
        raise DataValidationError(
            f"Need 80 valid distinct examples; found {len(selected)}"
        )
    for i, item in enumerate(selected):
        item["split"] = "train" if i < 64 else "dev"
    reviewed = sum(x["review_approved"] for x in selected)
    rendered = sum(
        isinstance(x["rendered_tokens"], int)
        and not isinstance(x["rendered_tokens"], bool)
        and 0 < x["rendered_tokens"] <= 1024
        and bool(x["renderer_revision"])
        for x in selected
    )
    manifest = {
        "kind": "training",
        "source": "allenai/wildguardmix",
        "configuration": "wildguardtrain",
        "revision": revision,
        "seed": seed,
        "train": selected[:64],
        "dev": selected[64:],
        "rejected": rejected,
        "eligible_unique_count": len(candidates),
        "ready": reviewed == 80 and rendered == 80,
        "review": {
            "approved_count": reviewed,
            "renderer_verified_count": rendered,
            "status": "complete" if reviewed == 80 and rendered == 80 else "pending",
        },
        "near_duplicate_method": "normalized words; Jaccard>=0.8 or character similarity>=0.9 with shared words; manual semantic audit still required",
    }
    manifest["manifest_hash"] = _hash(manifest)
    return manifest


def freeze_evaluation(suites: Mapping[str, Sequence[dict]], *, seed: int = 42) -> dict:
    """Freeze 89 pilot IDs and reserve all remaining items as untouched final."""
    rng = random.Random(seed)
    pilot, final, seen = [], [], set()
    expected_suites = {"harmbench", "mmlu", "xstest"}
    if set(suites) != expected_suites:
        raise DataValidationError("Require harmbench, mmlu, and xstest suites")
    for suite in sorted(suites):
        grouped = {}
        for original in sorted(suites[suite], key=lambda x: str(x.get("item_id", ""))):
            item = dict(original)
            if not item.get("item_id") or item["item_id"] in seen:
                raise DataValidationError("Missing or duplicate evaluation item ID")
            if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
                raise DataValidationError("Evaluation item has missing prompt")
            seen.add(item["item_id"])
            item.update(
                suite=suite,
                context=item.get("context", ""),
                expected_answer=item.get("expected_answer"),
                prompt_hash=_hash(normalize_prompt(item["prompt"])),
            )
            grouped.setdefault(item.get("category"), []).append(item)
        if suite == "harmbench":
            quotas = {"standard": 6, "contextual": 5, "copyright": 5}
        elif suite == "xstest":
            quotas = {"safe": 8, "unsafe": 8}
        else:
            if len(grouped) != 57:
                raise DataValidationError(
                    f"MMLU requires 57 subjects; found {len(grouped)}"
                )
            quotas = dict.fromkeys(sorted(grouped), 1)
        if set(grouped) != set(quotas):
            raise DataValidationError(f"{suite} categories differ from declared strata")
        for category, count in quotas.items():
            items = grouped[category]
            if len(items) <= count:
                raise DataValidationError(
                    f"{suite}/{category} needs {count} pilot items and untouched final items"
                )
            rng.shuffle(items)
            for index, item in enumerate(items):
                item["split"] = "pilot" if index < count else "final"
                (pilot if index < count else final).append(item)
    manifest = {
        "kind": "evaluation",
        "seed": seed,
        "pilot": pilot,
        "final": final,
        "counts": {
            "pilot": dict(Counter(i["suite"] for i in pilot)),
            "final": dict(Counter(i["suite"] for i in final)),
        },
    }
    manifest["manifest_hash"] = _hash(manifest)
    return manifest


def assert_disjoint(train: dict, evaluation: dict) -> None:
    """Reject IDs, exact normalized prompts, and lexical near-overlap.

    Run against the whole held-out corpus before training, not just pilot IDs.
    This conservative lexical check cannot prove semantic non-overlap.
    """
    training = train.get("train", []) + train.get("dev", [])
    held_out = evaluation.get("pilot", []) + evaluation.get("final", [])
    by_id = {str(x["item_id"]) for x in held_out}
    exact = {normalize_prompt(x["prompt"]) for x in held_out}
    index = {}
    normalized = []
    for i, item in enumerate(held_out):
        norm = normalize_prompt(item["prompt"])
        normalized.append(norm)
        for token in set(norm.split()):
            index.setdefault(token, set()).add(i)
    for item in training:
        norm = normalize_prompt(item["prompt"])
        if str(item["item_id"]) in by_id or norm in exact:
            raise DataValidationError(f"Evaluation overlap: {item['item_id']}")
        # Candidate retrieval from rare words avoids comparing every long prompt.
        tokens = set(norm.split())
        tokens = sorted(tokens, key=lambda w: (len(index.get(w, ())), -len(w)))
        candidates = set()
        for word in tokens[:8]:
            candidates.update(index.get(word, ()))
        for candidate in sorted(candidates):
            if _near(norm, normalized[candidate]):
                raise DataValidationError(
                    f"Evaluation near-overlap requires review: {item['item_id']}"
                )


def public_manifest(manifest: dict) -> dict:
    """Strip private dataset text from a manifest before publication review."""
    private_keys = {
        "prompt",
        "response",
        "context",
        "expected_answer",
        "question",
        "choices",
        "token_ids",
        "weights",
    }

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in private_keys}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(manifest)
