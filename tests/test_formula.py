import pytest

from qdrant_client.hybrid.formula import evaluate_variable, parse_variable


def test_parse_variable_score_index() -> None:
    assert parse_variable("$score") == 0
    assert parse_variable("$score[0]") == 0
    assert parse_variable("$score[2]") == 2
    assert parse_variable("$score[10]") == 10

    # qdrant core represents the score index as a usize, so anything that is not a plain
    # run of ascii digits is not a valid score pattern
    for var in ("$score[-1]", "$score[+1]", "$score[1_0]", "$score[ 1 ]", "$score[²]"):
        with pytest.raises(ValueError):
            parse_variable(var)


def test_evaluate_variable_rejects_negative_score_index() -> None:
    scores = [{1: 10.0}, {1: 20.0}, {1: 30.0}]

    assert evaluate_variable("$score[0]", 1, scores, {}, {}) == 10.0
    assert evaluate_variable("$score[2]", 1, scores, {}, {}) == 30.0
    # an index past the end falls back to the default score
    assert evaluate_variable("$score[3]", 1, scores, {}, {}) == 0.0

    # a negative index must not wrap around to the last prefetch, nor leak an IndexError
    with pytest.raises(ValueError):
        evaluate_variable("$score[-1]", 1, scores, {}, {})
    with pytest.raises(ValueError):
        evaluate_variable("$score[-9]", 1, scores, {}, {})
