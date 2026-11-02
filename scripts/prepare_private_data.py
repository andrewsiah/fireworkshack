"""Fetch pinned datasets into private storage and freeze pilot IDs.

Run explicitly after accepting the source access conditions. Successful gated
HF download verifies existing access; this script never accepts conditions.
No inference, training, uploads, or provider calls occur here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from fireworkshack.data import (
    DataValidationError,
    assert_disjoint,
    freeze_evaluation,
    normalize_prompt,
    public_manifest,
    select_training,
)

WG_REV = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"
HB_REV = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
XS_REV = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"


def save_private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def source_csv(url: str) -> tuple[list[dict], str]:
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.text))), hashlib.sha256(
        response.content
    ).hexdigest()


def prepare(
    output: Path,
    *,
    maximum_candidate_characters: int = 900,
    reviews_path: Path | None = None,
) -> dict:
    # Pinned revisions are immutable; no dataset-name fallback is permitted.
    wg_path = hf_hub_download(
        repo_id="allenai/wildguardmix",
        repo_type="dataset",
        revision=WG_REV,
        filename="train/wildguard_train.parquet",
    )
    mmlu_path = hf_hub_download(
        repo_id="cais/mmlu",
        repo_type="dataset",
        revision=MMLU_REV,
        filename="all/test-00000-of-00001.parquet",
    )
    wildguard = pq.read_table(wg_path).to_pylist()
    mmlu = pq.read_table(mmlu_path).to_pylist()
    harmbench, hb_hash = source_csv(
        f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{HB_REV}/data/behavior_datasets/harmbench_behaviors_text_test.csv"
    )
    xstest, xs_hash = source_csv(
        f"https://raw.githubusercontent.com/paul-rottger/xstest/{XS_REV}/xstest_prompts.csv"
    )
    suites = {
        "harmbench": [
            {
                "item_id": "harmbench:" + r["BehaviorID"],
                "suite": "harmbench",
                "prompt": r["Behavior"],
                "context": r["ContextString"],
                "category": r["FunctionalCategory"],
                "semantic_category": r["SemanticCategory"],
                "revision": HB_REV,
            }
            for r in harmbench
        ],
        "mmlu": [
            {
                "item_id": f"mmlu:{r['subject']}:{i}",
                "suite": "mmlu",
                "prompt": r["question"],
                "context": "",
                "choices": r["choices"],
                "expected_answer": "ABCD"[r["answer"]],
                "category": r["subject"],
                "revision": MMLU_REV,
            }
            for i, r in enumerate(mmlu)
        ],
        "xstest": [
            {
                "item_id": "xstest:" + str(r["id"]),
                "suite": "xstest",
                "prompt": r["prompt"],
                "context": "",
                "category": r["label"],
                "source_type": r["type"],
                "revision": XS_REV,
            }
            for r in xstest
        ],
    }
    evaluation = freeze_evaluation(suites)
    evaluation["sources"] = {
        "harmbench": {"revision": HB_REV, "sha256": hb_hash, "rows": len(harmbench)},
        "mmlu": {
            "revision": MMLU_REV,
            "sha256": hashlib.sha256(Path(mmlu_path).read_bytes()).hexdigest(),
            "rows": len(mmlu),
        },
        "xstest": {"revision": XS_REV, "sha256": xs_hash, "rows": len(xstest)},
    }
    # Exact full-corpus exclusion precedes deterministic selection. Near matches
    # are removed through assert_disjoint below, retaining every rejected ID.
    eval_norm = {
        normalize_prompt(r["prompt"]) for values in suites.values() for r in values
    }
    source_rejections = []
    reviews = json.loads(reviews_path.read_text()) if reviews_path else {}
    eligible = []
    for i, original in enumerate(wildguard):
        row = dict(original, item_id=f"wildguardtrain:{i}")
        if normalize_prompt(row.get("prompt") or "") in eval_norm:
            source_rejections.append(
                {"item_id": row["item_id"], "reason": "exact_full_evaluation_overlap"}
            )
        elif row.get("adversarial"):
            source_rejections.append(
                {"item_id": row["item_id"], "reason": "pilot_vanilla_only"}
            )
        elif (
            len(row.get("prompt") or "") + len(row.get("response") or "")
            > maximum_candidate_characters
        ):
            source_rejections.append(
                {
                    "item_id": row["item_id"],
                    "reason": "pilot_short_example_character_prefilter",
                }
            )
        elif re.search(
            r"\b(contact|telephone|phone|address|instagram|twitter|linkedin|social media|gps|coordinates|directions|license plate|net worth|bank balance|password|email|website|handle|fax|mailing|visiting hours|opening hours|business hours|admission fees|customer service|direct line|social security|license number|registration number)\b",
            row.get("prompt") or "",
            re.IGNORECASE,
        ):
            source_rejections.append(
                {
                    "item_id": row["item_id"],
                    "reason": "pilot_excludes_unverified_contact_or_dynamic_trivia",
                }
            )
        elif re.search(
            r"\[/?(?:ASS|CLIENT|AI|DOC|INST)|<\||^\s*\[[A-Z_/]{2,16}\]",
            row.get("response") or "",
            re.IGNORECASE,
        ):
            source_rejections.append(
                {"item_id": row["item_id"], "reason": "role_markup_in_source_response"}
            )
        elif reviews.get(row["item_id"], {}).get("approved") is False:
            source_rejections.append(
                {"item_id": row["item_id"], "reason": "semantic_audit_rejected"}
            )
        else:
            eligible.append(row)
    from fireworkshack.train import NativeRenderer, prepare_training_data

    renderer = NativeRenderer.load(local_files_only=True)
    for _attempt in range(200):
        training = select_training(eligible, revision=WG_REV)
        oversized = [
            row["item_id"]
            for split in ["train", "dev"]
            for row in training[split]
            if renderer.render_supervised(row["prompt"], row["response"])[
                "rendered_tokens"
            ]
            > 1024
        ]
        if oversized:
            source_rejections.extend(
                {"item_id": identifier, "reason": "native_rendered_length_exceeds_1024"}
                for identifier in oversized
            )
            eligible = [row for row in eligible if row["item_id"] not in oversized]
            continue
        try:
            assert_disjoint(training, evaluation)
            break
        except DataValidationError as exc:
            identifier = str(exc).split(": ", 1)[1]
            if not identifier.startswith("wildguardtrain:"):
                raise
            source_rejections.append(
                {
                    "item_id": identifier,
                    "reason": "lexical_full_evaluation_near_overlap",
                }
            )
            eligible = [r for r in eligible if r["item_id"] != identifier]
    else:
        raise DataValidationError(
            "Overlap screening exceeded bounded reselection attempts"
        )
    training["source_rejections"] = source_rejections
    training["source_metadata"] = {
        "gated_access": "verified_by_successful_download",
        "rows": len(wildguard),
        "sha256": hashlib.sha256(Path(wg_path).read_bytes()).hexdigest(),
        "manual_semantic_review": "pending",
        "renderer_token_limits": "verified_with_pinned_native_encoder",
        "candidate_policy": {
            "vanilla_only": True,
            "maximum_candidate_characters": maximum_candidate_characters,
            "note": "Short, non-adversarial benign-response workflow pilot; narrower than general overrefusal training.",
        },
    }
    for item in training["train"] + training["dev"]:
        review = reviews.get(item["item_id"], {})
        item["review_approved"] = (
            review.get("approved") is True
            and review.get("content_hash") == item["content_hash"]
        )
        if item["review_approved"]:
            item["reviewer"] = review.get("reviewer", "unspecified")
    training = prepare_training_data(training, renderer)
    reviewed = sum(
        item["review_approved"]
        for split in ["train", "dev"]
        for item in training[split]
    )
    training["review"] = {
        "approved_count": reviewed,
        "renderer_verified_count": 80,
        "status": "complete" if reviewed == 80 else "pending",
    }
    training["source_metadata"]["manual_semantic_review"] = (
        "complete" if reviewed == 80 else "pending"
    )
    # Include preparation metadata in the final hashes rather than retaining
    # hashes from the earlier in-memory selection phase.
    for manifest in [training, evaluation]:
        manifest.pop("manifest_hash", None)
        encoded = json.dumps(
            manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        manifest["manifest_hash"] = hashlib.sha256(encoded).hexdigest()
    save_private(output / "training.json", training)
    save_private(output / "evaluation.json", evaluation)
    save_private(output / "public-training-candidate.json", public_manifest(training))
    save_private(
        output / "public-evaluation-candidate.json", public_manifest(evaluation)
    )
    summary = {
        "access": "verified",
        "training_selected": len(training["train"]),
        "development_selected": len(training["dev"]),
        "training_ready": training["ready"],
        "training_revision": WG_REV,
        "evaluation_counts": evaluation["counts"],
        "full_corpus_overlap_check": "passed",
        "source_overlap_exclusions": sum(
            "overlap" in rejection["reason"] for rejection in source_rejections
        ),
        "source_prefilter_exclusions": len(source_rejections),
        "train_tokens": training["train_tokens"],
        "max_rendered_tokens": training["max_rendered_tokens"],
        "reviewed_examples": reviewed,
        "training_manifest_hash": training["manifest_hash"],
        "evaluation_manifest_hash": evaluation["manifest_hash"],
        "manifests": str(output),
        "paid_requests": 0,
        "remaining": []
        if training["ready"]
        else ["manual semantic review of selected training rows"],
    }
    save_private(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--maximum-candidate-characters", type=int, default=900)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                maximum_candidate_characters=args.maximum_candidate_characters,
                reviews_path=args.reviews,
            ),
            indent=2,
        )
    )
