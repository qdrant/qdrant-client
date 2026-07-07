from qdrant_client import models
from qdrant_client.local.payload_filters import check_match


def test_match_text_on_non_string_value_does_not_crash():
    # Mixed payload types on the same key must not crash MatchText / MatchTextAny.
    # https://github.com/qdrant/qdrant-client/issues/1221
    assert check_match(models.MatchText(text="42"), 42) is False
    assert check_match(models.MatchTextAny(text_any="42"), 42) is False
    assert check_match(models.MatchText(text="x"), None) is False
    assert check_match(models.MatchTextAny(text_any="x"), None) is False


def test_match_text_still_matches_strings():
    assert check_match(models.MatchText(text="ell"), "hello") is True
    assert check_match(models.MatchText(text="zzz"), "hello") is False
    assert check_match(models.MatchTextAny(text_any="foo hello"), "hello world") is True
    assert check_match(models.MatchTextAny(text_any="foo bar"), "hello world") is False
