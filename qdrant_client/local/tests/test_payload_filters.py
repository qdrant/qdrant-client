from qdrant_client.http.models import models
from qdrant_client.local.payload_filters import check_filter, check_match


def test_nested_payload_filters():
    payload = {
        "country": {
            "name": "Germany",
            "capital": "Berlin",
            "cities": [
                {
                    "name": "Berlin",
                    "population": 3.7,
                    "location": {
                        "lon": 13.76116,
                        "lat": 52.33826,
                    },
                    "sightseeing": ["Brandenburg Gate", "Reichstag"],
                },
                {
                    "name": "Munich",
                    "population": 1.5,
                    "location": {
                        "lon": 11.57549,
                        "lat": 48.13743,
                    },
                    "sightseeing": ["Marienplatz", "Olympiapark"],
                },
                {
                    "name": "Hamburg",
                    "population": 1.8,
                    "location": {
                        "lon": 9.99368,
                        "lat": 53.55108,
                    },
                    "sightseeing": ["Reeperbahn", "Elbphilharmonie"],
                },
            ],
        }
    }

    query = models.Filter(
        **{
            "must": [
                {
                    "nested": {
                        "key": "country.cities",
                        "filter": {
                            "must": [
                                {
                                    "key": "population",
                                    "range": {
                                        "gte": 1.0,
                                    },
                                }
                            ],
                            "must_not": [{"key": "sightseeing", "values_count": {"gt": 1}}],
                        },
                    }
                }
            ]
        }
    )

    res = check_filter(query, payload, 0, has_vector={})
    assert res is False

    query = models.Filter(
        **{
            "must": [
                {
                    "nested": {
                        "key": "country.cities",
                        "filter": {
                            "must": [
                                {
                                    "key": "population",
                                    "range": {
                                        "gte": 1.0,
                                    },
                                }
                            ]
                        },
                    }
                }
            ]
        }
    )

    res = check_filter(query, payload, 0, has_vector={})
    assert res is True

    query = models.Filter(
        **{
            "must": [
                {
                    "nested": {
                        "key": "country.cities",
                        "filter": {
                            "must": [
                                {
                                    "key": "population",
                                    "range": {
                                        "gte": 1.0,
                                    },
                                },
                                {"key": "sightseeing", "values_count": {"gt": 2}},
                            ]
                        },
                    }
                }
            ]
        }
    )

    res = check_filter(query, payload, 0, has_vector={})
    assert res is False

    query = models.Filter(
        **{
            "must": [
                {
                    "nested": {
                        "key": "country.cities",
                        "filter": {
                            "must": [
                                {
                                    "key": "population",
                                    "range": {
                                        "gte": 9.0,
                                    },
                                }
                            ]
                        },
                    }
                }
            ]
        }
    )

    res = check_filter(query, payload, 0, has_vector={})
    assert res is False


def test_geo_polygon_filter_query():
    payload = {
        "location": [
            {
                "lon": 70.0,
                "lat": 70.0,
            },
        ]
    }

    query = models.Filter(
        **{
            "must": [
                {
                    "key": "location",
                    "geo_polygon": {
                        "exterior": {
                            "points": [
                                {"lon": 55.455868, "lat": 55.495862},
                                {"lon": 86.455868, "lat": 55.495862},
                                {"lon": 86.455868, "lat": 86.495862},
                                {"lon": 55.455868, "lat": 86.495862},
                                {"lon": 55.455868, "lat": 55.495862},
                            ]
                        },
                    },
                }
            ]
        }
    )

    res = check_filter(query, payload, 0, has_vector={})
    assert res is True

    payload = {
        "location": [
            {
                "lon": 30.693738,
                "lat": 30.502165,
            },
        ]
    }

    res = check_filter(query, payload, 0, has_vector={})
    assert res is False


def text(query: str) -> models.MatchText:
    return models.MatchText(text=query)


def phrase(query: str) -> models.MatchPhrase:
    return models.MatchPhrase(phrase=query)


def test_text_match_uses_token_matching_not_substring():
    """On a field without a text index the server matches whole tokens, not substrings
    (qdrant#10341). Cases mirror the server's own `unindexed_text_match_test.rs`.
    """
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


def test_phrase_match_requires_token_order():
    assert check_match(phrase("alpha beta"), "foo alpha beta bar")
    assert not check_match(phrase("alpha beta"), "beta alpha")
    assert not check_match(phrase("alpha beta"), "alphabeta")
    # consecutive, not merely in order: an ordered subsequence is not a phrase
    assert not check_match(phrase("alpha beta"), "alpha x beta")
    assert not check_match(phrase("good"), "goodness only")
    assert check_match(phrase("good"), "goodness only good")

    assert check_match(phrase("Alpha, Beta!"), "alpha beta")
    assert not check_match(phrase(""), "anything")
    assert not check_match(phrase("alpha"), None)


def test_text_any_match_keeps_substring_semantics():
    """Unlike text and phrase, the server still resolves `MatchTextAny` on an unindexed
    field with a substring scan.
    """
    assert check_match(models.MatchTextAny(text_any="good fly"), "goodness only")
    assert check_match(models.MatchTextAny(text_any="fly"), "butterfly")
    assert not check_match(models.MatchTextAny(text_any="cheap"), "goodness only")
