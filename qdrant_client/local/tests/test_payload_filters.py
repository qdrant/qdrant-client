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


def test_values_count_with_dict_payload():
    """Dict payloads should count as 1 scalar value, not len(dict) keys."""
    payload = {
        "metadata": {"key1": "val1", "key2": "val2"},
    }

    # A dict with 2 keys should count as 1 (scalar JSON object), not 2
    query_eq_1 = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata",
                values_count=models.ValuesCount(gte=1, lte=1),
            )
        ]
    )
    assert check_filter(query_eq_1, payload, 0, has_vector={}) is True

    # Should NOT match count >= 2, since dict is a single value
    query_gte_2 = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata",
                values_count=models.ValuesCount(gte=2),
            )
        ]
    )
    assert check_filter(query_gte_2, payload, 0, has_vector={}) is False

    # gt and lt operators should also work correctly with dict payloads
    query_gt_0 = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata",
                values_count=models.ValuesCount(gt=0),
            )
        ]
    )
    assert check_filter(query_gt_0, payload, 0, has_vector={}) is True

    query_lt_2 = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata",
                values_count=models.ValuesCount(lt=2),
            )
        ]
    )
    assert check_filter(query_lt_2, payload, 0, has_vector={}) is True

    # Empty dict should count as 1 (a value exists), not 0
    empty_dict_payload = {"metadata": {}}
    assert check_filter(query_eq_1, empty_dict_payload, 0, has_vector={}) is True

    # Nested dict should also count as 1
    nested_dict_payload = {"metadata": {"outer": {"inner": "value"}}}
    assert check_filter(query_eq_1, nested_dict_payload, 0, has_vector={}) is True

    # A list of 3 items should still count as 3
    list_payload = {"tags": ["a", "b", "c"]}
    query_eq_3 = models.Filter(
        must=[
            models.FieldCondition(
                key="tags",
                values_count=models.ValuesCount(gte=3, lte=3),
            )
        ]
    )
    assert check_filter(query_eq_3, list_payload, 0, has_vector={}) is True


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
