"""Local mode must mirror the server's token-aware text/phrase matching on
unindexed payload fields (qdrant/qdrant#10341), which replaced the old
substring scan."""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.local.payload_filters import check_match


def text(query: str) -> models.MatchText:
    return models.MatchText(text=query)


def phrase(query: str) -> models.MatchPhrase:
    return models.MatchPhrase(phrase=query)


def test_text_uses_token_matching_not_substring() -> None:
    # cases from the server's unindexed_text_match_test.rs
    assert not check_match(text("good"), "goodness only")
    assert check_match(text("good"), "good cheap stuff")
    assert check_match(text("good cheap"), "cheap hardware good")
    assert not check_match(text("good cheap"), "cheap hardware")

    # tokenization: split on non-alphanumeric, lowercase
    assert check_match(text("FLY"), "fly agaric")
    assert check_match(text("fly"), "come fly, with me")
    assert not check_match(text("fly"), "butterfly dragonfly")
    assert not check_match(text(""), "anything")
    assert not check_match(text("fly"), 7)


def test_phrase_requires_token_order() -> None:
    # cases from the server's unindexed_text_match_test.rs
    assert check_match(phrase("alpha beta"), "foo alpha beta bar")
    assert not check_match(phrase("alpha beta"), "beta alpha")
    assert not check_match(phrase("alpha beta"), "alphabeta")
    assert not check_match(phrase("good"), "goodness only")
    assert check_match(phrase("good"), "goodness only good")

    assert check_match(phrase("Alpha, Beta!"), "alpha beta")
    assert not check_match(phrase(""), "anything")
    assert not check_match(phrase("alpha"), None)


def test_text_any_keeps_substring_semantics() -> None:
    # the server still resolves MatchTextAny with a substring scan
    assert check_match(models.MatchTextAny(text_any="good fly"), "goodness only")


def test_local_client_count_with_text_filter() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    client.upsert(
        "test",
        points=[
            models.PointStruct(id=i, vector=[0.0, 1.0], payload={"words": words})
            for i, words in enumerate(["butterfly frog", "fly agaric", "dragonfly elephant"])
        ],
    )
    count_filter = models.Filter(
        must=[models.FieldCondition(key="words", match=models.MatchText(text="fly"))]
    )
    assert client.count("test", count_filter=count_filter).count == 1
