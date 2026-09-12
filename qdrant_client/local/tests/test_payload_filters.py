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


def test_text_any_match_needs_only_one_token():
    """Like text and phrase, `MatchTextAny` on an unindexed field matches whole tokens
    rather than substrings (qdrant#10526), but one query token is enough.
    """
    text_any = models.MatchTextAny

    # a substring of a document token is not a match
    assert not check_match(text_any(text_any="good fly"), "goodness only")
    assert not check_match(text_any(text_any="fly"), "butterfly")
    assert not check_match(text_any(text_any="cheap"), "goodness only")

    # any single query token is enough, unlike `MatchText`, which requires all of them
    assert check_match(text_any(text_any="good fly"), "good cheap stuff")
    assert check_match(text_any(text_any="good fly"), "come fly, with me")
    assert not check_match(text_any(text_any="good fly"), "cheap hardware")

    # tokenization applies to both sides, lowercasing included
    assert check_match(text_any(text_any="Alpha, Beta!"), "beta")
    assert not check_match(text_any(text_any=""), "anything")
    assert not check_match(text_any(text_any="alpha"), None)


def matching_ids(flt: models.Filter, payloads: dict) -> list:
    return [
        idx for idx, payload in payloads.items() if check_filter(flt, payload, idx, has_vector={})
    ]


EMPTY_NULL_PAYLOADS = {
    1: {"reports": [1, 2]},  # non-empty array: neither empty nor null
    2: {"reports": []},  # empty array: empty, not null
    3: {"reports": None},  # null: both empty and null
    4: {},  # key absent: empty, not null
    5: {"reports": [None, 1]},  # array holding a null: not empty, but null
}


def is_empty_null_filter(flag: str, value: bool) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="reports", **{flag: value})])


def test_field_condition_is_empty():
    # `FieldCondition.is_empty` is the shorthand syntax for `IsEmptyCondition`, and on a key
    # holding a single value the two agree.
    assert matching_ids(is_empty_null_filter("is_empty", True), EMPTY_NULL_PAYLOADS) == [2, 3, 4]
    assert matching_ids(is_empty_null_filter("is_empty", False), EMPTY_NULL_PAYLOADS) == [1, 5]

    negated = models.Filter(must_not=[models.FieldCondition(key="reports", is_empty=True)])
    assert matching_ids(negated, EMPTY_NULL_PAYLOADS) == [1, 5]

    verbose = models.Filter(
        must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="reports"))]
    )
    assert matching_ids(verbose, EMPTY_NULL_PAYLOADS) == [2, 3, 4]


def test_field_condition_is_null():
    # `FieldCondition.is_null` matches a null value, or an array containing one. An absent
    # key is not null.
    assert matching_ids(is_empty_null_filter("is_null", True), EMPTY_NULL_PAYLOADS) == [3, 5]
    assert matching_ids(is_empty_null_filter("is_null", False), EMPTY_NULL_PAYLOADS) == [1, 2, 4]

    negated = models.Filter(must_not=[models.FieldCondition(key="reports", is_null=True)])
    assert matching_ids(negated, EMPTY_NULL_PAYLOADS) == [1, 2, 4]

    verbose = models.Filter(
        must=[models.IsNullCondition(is_null=models.PayloadField(key="reports"))]
    )
    assert matching_ids(verbose, EMPTY_NULL_PAYLOADS) == [3, 5]


def test_field_condition_is_empty_is_null_json_path():
    # `IsEmptyCondition`
    # needs every value to be empty, so it parts company with `is_empty=True` exactly on the
    # points holding values of both kinds - agreeing everywhere else, single-valued keys
    # included, as `test_field_condition_is_empty` asserts.
    payloads = {
        1: {"a": [{"b": 1}, {"b": None}]},
        2: {"a": [{"b": 1}, {"b": 2}]},
        3: {"a": []},
        4: {"a": [{"b": []}, {"b": 1}]},
        5: {"a": [{"b": []}]},
    }

    def matches(flag: str, value: bool) -> list:
        return matching_ids(
            models.Filter(must=[models.FieldCondition(key="a[].b", **{flag: value})]), payloads
        )

    assert matches("is_null", True) == [1]
    # point 1 also holds a non-null value, so it satisfies both directions
    assert matches("is_null", False) == [1, 2, 3, 4, 5]
    assert matches("is_empty", True) == [1, 3, 4, 5]
    # likewise points 1 and 4 hold a non-empty value as well
    assert matches("is_empty", False) == [1, 2, 4]

    verbose_empty = models.Filter(
        must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="a[].b"))]
    )
    assert matching_ids(verbose_empty, payloads) == [3, 5]

    verbose_null = models.Filter(
        must=[models.IsNullCondition(is_null=models.PayloadField(key="a[].b"))]
    )
    assert matching_ids(verbose_null, payloads) == matches("is_null", True)
