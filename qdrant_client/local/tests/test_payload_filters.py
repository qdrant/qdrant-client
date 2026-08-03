from qdrant_client.http.models import models
from qdrant_client.local.payload_filters import check_filter


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


def test_field_condition_is_empty():
    # `FieldCondition.is_empty` is the shorthand syntax for `IsEmptyCondition`.
    # A value is empty if it is null, an empty array, or the key is absent.
    payloads = {
        1: {"reports": [1, 2]},
        2: {"reports": []},
        3: {"reports": None},
        4: {},
        5: {"reports": [None, 1]},
    }

    def matches(is_empty: bool) -> list[int]:
        query = models.Filter(must=[models.FieldCondition(key="reports", is_empty=is_empty)])
        return [
            idx
            for idx, payload in payloads.items()
            if check_filter(query, payload, idx, has_vector={})
        ]

    assert matches(True) == [2, 3, 4]
    assert matches(False) == [1, 5]

    # `must_not` must be the exact complement, not "everything"
    negated = models.Filter(must_not=[models.FieldCondition(key="reports", is_empty=True)])
    assert [
        idx
        for idx, payload in payloads.items()
        if check_filter(negated, payload, idx, has_vector={})
    ] == [1, 5]

    # for a key holding a single value, the shorthand agrees with the verbose condition
    # it abbreviates
    verbose = models.Filter(
        must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="reports"))]
    )
    assert [
        idx
        for idx, payload in payloads.items()
        if check_filter(verbose, payload, idx, has_vector={})
    ] == matches(True)


def test_field_condition_is_null():
    # `FieldCondition.is_null` matches a null value, or an array containing one.
    # An absent key is not null.
    payloads = {
        1: {"reports": [1, 2]},
        2: {"reports": []},
        3: {"reports": None},
        4: {},
        5: {"reports": [None, 1]},
    }

    def matches(is_null: bool) -> list[int]:
        query = models.Filter(must=[models.FieldCondition(key="reports", is_null=is_null)])
        return [
            idx
            for idx, payload in payloads.items()
            if check_filter(query, payload, idx, has_vector={})
        ]

    assert matches(True) == [3, 5]
    assert matches(False) == [1, 2, 4]

    negated = models.Filter(must_not=[models.FieldCondition(key="reports", is_null=True)])
    assert [
        idx
        for idx, payload in payloads.items()
        if check_filter(negated, payload, idx, has_vector={})
    ] == [1, 2, 4]

    # Deliberately no equivalence assertion against `IsNullCondition` here. On a field
    # without a payload index the server does not treat the two as interchangeable: the
    # verbose condition tests the values the key resolves to, so an array holding a null
    # is not itself null, while the shorthand looks inside it. Point 5 matches the
    # shorthand only. Local mode does not model payload indexes, so it mirrors that.
    verbose = models.Filter(
        must=[models.IsNullCondition(is_null=models.PayloadField(key="reports"))]
    )
    assert [
        idx
        for idx, payload in payloads.items()
        if check_filter(verbose, payload, idx, has_vector={})
    ] == [3]


def test_field_condition_is_empty_is_null_with_values_count():
    # a condition carrying values_count as well keeps going to the values_count branch,
    # so adding is_empty / is_null to it changes nothing
    payloads = {
        1: {"reports": [1, 2]},
        2: {"reports": []},
        3: {"reports": None},
        4: {},
    }

    def matches(query: models.Filter) -> list[int]:
        return [
            idx
            for idx, payload in payloads.items()
            if check_filter(query, payload, idx, has_vector={})
        ]

    assert matches(
        models.Filter(
            must=[
                models.FieldCondition(
                    key="reports", is_empty=True, values_count=models.ValuesCount(gte=1)
                )
            ]
        )
    ) == [1]
    assert matches(
        models.Filter(
            must=[
                models.FieldCondition(
                    key="reports", is_null=True, values_count=models.ValuesCount(lt=1)
                )
            ]
        )
    ) == [2, 3, 4]


def test_field_condition_is_empty_is_null_json_path():
    # a key resolving to several values matches when any one of them satisfies the
    # condition
    payloads = {
        1: {"a": [{"b": 1}, {"b": None}]},
        2: {"a": [{"b": 1}, {"b": 2}]},
        3: {"a": []},
        4: {"a": [{"b": []}, {"b": 1}]},
    }

    def matches(kind: str, flag: bool) -> list[int]:
        query = models.Filter(must=[models.FieldCondition(key="a[].b", **{kind: flag})])
        return [
            idx
            for idx, payload in payloads.items()
            if check_filter(query, payload, idx, has_vector={})
        ]

    assert matches("is_null", True) == [1]
    # point 1 also holds a non-null value, so it satisfies both directions
    assert matches("is_null", False) == [1, 2, 3, 4]
    assert matches("is_empty", True) == [1, 3, 4]
    # likewise points 1 and 4 hold a non-empty value as well
    assert matches("is_empty", False) == [1, 2, 4]
