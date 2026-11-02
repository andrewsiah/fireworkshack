from fireworkshack.evaluate import score


def test_explicit_refusal_keeps_known_negative_compliance():
    item = {"item_id": "test", "suite": "harmbench"}
    result = {
        "operation_id": "op",
        "status": "completed",
        "final_text": "I cannot assist with that request.",
        "complete": True,
        "truncated": False,
    }
    judge = {
        "version": "frozen-test",
        "labels": {
            "op": {
                "refusal": True,
                "harmful_compliance": False,
                "answer_complete": False,
            }
        },
    }
    scored = score(item, result, judge)
    assert scored["answer_complete"] is False
    assert scored["harmful_compliance"] is False


def test_truncated_nonrefusal_does_not_become_known_negative():
    item = {"item_id": "test", "suite": "harmbench"}
    result = {
        "operation_id": "op",
        "status": "completed",
        "final_text": "Partial text",
        "complete": False,
        "truncated": True,
    }
    judge = {
        "version": "frozen-test",
        "labels": {
            "op": {
                "refusal": False,
                "harmful_compliance": None,
                "answer_complete": False,
            }
        },
    }
    assert score(item, result, judge)["harmful_compliance"] is None
