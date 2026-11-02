import hashlib
import json

import pytest

from fireworkshack.data import (
    DataValidationError,
    assert_disjoint,
    freeze_evaluation,
    public_manifest,
    select_training,
)


def row(i, **extra):
    text = hashlib.sha256(str(i).encode()).hexdigest()
    return dict(
        item_id=f"w:{i}",
        prompt=f"Benign task {text}",
        response=f"Helpful response {text}",
        prompt_harm_label="unharmful",
        response_harm_label="unharmful",
        response_refusal_label="compliance",
        **extra,
    )


def rows(n=80):
    return [row(i) for i in range(n)]


def suites():
    return {
        "harmbench": [
            {
                "item_id": f"h:{c}:{i}",
                "suite": "harmbench",
                "prompt": f"{c} {i}",
                "context": "context survives",
                "category": c,
            }
            for c in ["standard", "contextual", "copyright"]
            for i in range(8)
        ],
        "mmlu": [
            {
                "item_id": f"m:{s}:{i}",
                "suite": "mmlu",
                "prompt": f"subject {s} question {i}",
                "category": f"subject_{s}",
                "expected_answer": "A",
            }
            for s in range(57)
            for i in range(2)
        ],
        "xstest": [
            {
                "item_id": f"x:{c}:{i}",
                "suite": "xstest",
                "prompt": f"{c} contrast {i}",
                "category": c,
            }
            for c in ["safe", "unsafe"]
            for i in range(10)
        ],
    }


def test_selection_filters_harmful_refusal_empty_and_is_repeatable():
    candidates = rows(85)
    candidates[0]["prompt_harm_label"] = "harmful"
    candidates[1]["response_harm_label"] = "harmful"
    candidates[2]["response_refusal_label"] = "refusal"
    candidates[3]["response"] = None
    candidates[4]["prompt_harm_label"] = None
    a = select_training(candidates, revision="a" * 40)
    b = select_training(candidates, revision="a" * 40)
    assert a == b
    assert len(a["train"]) == 64 and len(a["dev"]) == 16
    assert len(a["rejected"]) == 5
    assert not a["ready"]
    assert {i["item_id"] for i in a["train"]}.isdisjoint(i["item_id"] for i in a["dev"])


def test_under_80_valid_rows_stops():
    with pytest.raises(DataValidationError, match="80"):
        select_training(rows(79), revision="a" * 40)


def test_duplicate_ids_and_normalized_prompt_removed():
    candidates = rows(82)
    candidates[-2]["prompt"] = candidates[0]["prompt"].upper() + "  "
    candidates[-1]["item_id"] = candidates[1]["item_id"]
    result = select_training(candidates, revision="a" * 40)
    assert len(result["rejected"]) == 2


def test_evaluation_strata_context_and_final_reserved():
    data = freeze_evaluation(suites())
    assert len(data["pilot"]) == 89
    hb = [i for i in data["pilot"] if i["suite"] == "harmbench"]
    assert [
        sum(i["category"] == c for i in hb)
        for c in ["standard", "contextual", "copyright"]
    ] == [6, 5, 5]
    assert all(i["context"] == "context survives" for i in hb)
    assert sum(i["suite"] == "mmlu" for i in data["pilot"]) == 57
    assert [
        sum(i["category"] == c and i["suite"] == "xstest" for i in data["pilot"])
        for c in ["safe", "unsafe"]
    ] == [8, 8]
    assert {i["item_id"] for i in data["pilot"]}.isdisjoint(
        i["item_id"] for i in data["final"]
    )
    assert data == freeze_evaluation(suites())
    assert json.loads(json.dumps(data)) == data


def test_missing_subject_or_stratum_stops():
    data = suites()
    data["mmlu"] = [i for i in data["mmlu"] if i["category"] != "subject_0"]
    with pytest.raises(DataValidationError, match="57"):
        freeze_evaluation(data)


def test_overlap_and_near_duplicate_prompt_block():
    train = select_training(rows(), revision="a" * 40)
    example = train["train"][0]
    for prompt in [example["prompt"].upper(), "Please " + example["prompt"]]:
        evaluation = {"pilot": [{"item_id": "other", "prompt": prompt}], "final": []}
        with pytest.raises(DataValidationError, match="overlap|near"):
            assert_disjoint(train, evaluation)


def test_public_manifest_never_contains_dataset_text():
    train = select_training(rows(), revision="a" * 40)
    public = public_manifest(train)
    assert all("prompt" not in i and "response" not in i for i in public["train"])
    assert all("content_hash" in i for i in public["train"])


def test_readiness_needs_both_review_and_real_rendered_lengths():
    candidates = rows()
    for item in candidates:
        item.update(
            review_approved=True,
            rendered_tokens=100,
            renderer_revision="renderer@pinned",
        )
    assert select_training(candidates, revision="a" * 40)["ready"]
    candidates[0]["rendered_tokens"] = 1025
    assert not select_training(candidates, revision="a" * 40)["ready"]


def test_final_confirmation_overlap_is_also_rejected():
    train = select_training(rows(), revision="a" * 40)
    evaluation = {
        "pilot": [],
        "final": [{"item_id": "other", "prompt": train["dev"][0]["prompt"]}],
    }
    with pytest.raises(DataValidationError, match="overlap"):
        assert_disjoint(train, evaluation)


def test_public_manifest_strips_reversible_token_data():
    manifest = {
        "train": [
            {
                "item_id": "a",
                "content_hash": "safe",
                "token_ids": [1, 2, 3],
                "weights": [0.0, 1.0, 1.0],
                "rendered_tokens": 3,
            }
        ]
    }
    result = public_manifest(manifest)
    assert "token_ids" not in result["train"][0]
    assert "weights" not in result["train"][0]
    assert result["train"][0]["rendered_tokens"] == 3
