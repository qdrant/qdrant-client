import random
from datetime import date, datetime, timedelta, timezone
from typing import Union

from qdrant_client.http import models
from tests.fixtures.payload import geo_points, random_real_word, random_signed_int

"""
data structure:

{
  "city": {
    "geo": {
      "lat": 34.052235,
      "lon": -118.243683
    },
    "name": "Los Angeles"
  },
  "id": 101,
  "id_str": [ "28", "02", "17", "11", "21" ],
  "nested": {
    "array": [
      {
        "number": 5,
        "word": "butterfly"
      },
      {
        "number": 3,
        "word": "tiger"
      }
    ],
    "id": 101
  },
  "nested_array": [[1, 8], [-5, 9, 3]],
  "rand_digit": 3,
  "rand_signed_int": -3,
  "rand_datetime": "2021-08-17T14:00:00+00:00",  # or datetime object
  "rand_number": 0.8558,
  "text_array": [
    "3044671ce68848fb839a3ae8cb523cb8",
    "44c89ac703a3415db8b0e88b5bdd88d3"
  ],
  "text_data": "6d610fbd8a1345b4a8bcb4161a0a3a52",
  "words": "ape wolf",
  "two_words": ["ape", "wolf"],
  "maybe": None,
}

Possible filters:

- IsEmptyCondition
- IsNullCondition
- FieldCondition
    - match
        - value
        - text
        - number
        - any(in)
    - range
    - geo_bounding_box
    - geo_radius
    - values_count
- HasIdCondition

"""


def is_empty_condition() -> models.IsEmptyCondition:
    return models.IsEmptyCondition(
        is_empty=models.PayloadField(
            key=random.choice(
                ["maybe", "nested.array[].nested_empty", "nested.array[].nested_empty2"]
            )
        )
    )


def is_null_condition() -> models.IsNullCondition:
    return models.IsNullCondition(is_null=models.PayloadField(key="maybe_null"))


def has_id_condition(num_ids: int = 10, max_id: int = 1000) -> models.HasIdCondition:
    random_ids = [random.randint(1, max_id) for _ in range(num_ids)]
    return models.HasIdCondition(has_id=random_ids)


def nested_field_condition_1() -> models.Condition:
    value = random_real_word()
    lt = random.randint(1, 10)

    return models.NestedCondition(
        nested=models.Nested(
            key="nested.array",
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="word",
                        match=models.MatchValue(value=value),
                    ),
                    models.FieldCondition(
                        key="number",
                        range=models.Range(lt=lt),
                    ),
                ]
            ),
        )
    )


def nested_field_condition_2() -> models.Condition:
    value = random_real_word()
    lt = random.randint(1, 10)

    return models.NestedCondition(
        nested=models.Nested(
            key="nested.array",
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="word",
                        match=models.MatchValue(value=value),
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="number",
                        range=models.Range(lt=lt),
                    )
                ],
            ),
        )
    )


def match_value_field_condition() -> models.FieldCondition:
    field = random.choice(["maybe", "nested.array[].word", "id_str"])

    if field in ["maybe", "nested.array[].word"]:
        value = random_real_word()
    else:
        value = str(random.randint(1, 30)).zfill(2)

    return models.FieldCondition(
        key=field,
        match=models.MatchValue(value=value),
    )


def match_text_field_condition() -> models.FieldCondition:
    field = "words"
    text = random_real_word()

    return models.FieldCondition(
        key=field,
        match=models.MatchText(text=text),
    )


def match_any_field_condition() -> models.FieldCondition:
    field = "id_str"
    any_vals = [str(random.randint(1, 30)).zfill(2) for _ in range(3)]

    return models.FieldCondition(
        key=field,
        match=models.MatchAny(any=any_vals),
    )


def match_except_field_condition() -> models.FieldCondition:
    field = "two_words"
    except_vals = [str(random.randint(1, 30)).zfill(2) for _ in range(10)]

    return models.FieldCondition(
        key=field,
        match=models.MatchExcept(**{"except": except_vals}),
    )


def range_field_condition() -> models.FieldCondition:
    field = "rand_number"
    lt = random.random()
    gt = random.random()

    if lt > gt:
        if lt > 0.5:
            lt = None
        else:
            gt = None

    return models.FieldCondition(
        key=field,
        range=models.Range(lt=lt, gt=gt),
    )


def datetime_range_field_condition() -> models.FieldCondition:
    field = "rand_datetime"

    start_datetime = datetime(2000, 1, 1)
    end_datetime = datetime(2001, 1, 31)

    def random_datetime() -> Union[datetime, date]:
        dt = start_datetime + timedelta(
            seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds())),
            microseconds=random.randint(0, 999999),
        )
        if random.random() > 0.8:
            return dt.date()

        return dt.replace(tzinfo=timezone(offset=timedelta(hours=random.randint(-12, 12))))

    lt = random_datetime()
    gt = random_datetime()

    rand_1 = random.random()
    rand_2 = random.random()

    if rand_1 > rand_2:
        if rand_1 > 0.5:
            lt = None
        else:
            gt = None

    return models.FieldCondition(
        key=field,
        range=models.DatetimeRange(lt=lt, gt=gt),
    )


def range_nested_array_field_condition() -> models.Condition:
    first_index, second_index = random.choices([0, 1, 2, 10, ""], k=2)
    lt = random_signed_int()

    return models.FieldCondition(
        key=f"nested_array[{first_index}][{second_index}]",
        range=models.Range(lt=lt),
    )


def geo_bounding_box_field_condition() -> models.FieldCondition:
    field = "city.geo"
    random_top_left = {
        "lat": random.random() * 180 - 90,
        "lon": random.random() * 360 - 180,
    }
    random_bottom_right = {
        "lat": random.random() * 180 - 90,
        "lon": random.random() * 360 - 180,
    }

    return models.FieldCondition(
        key=field,
        geo_bounding_box=models.GeoBoundingBox(
            top_left=random_top_left, bottom_right=random_bottom_right
        ),
    )


def geo_radius_field_condition() -> models.FieldCondition:
    field = "city.geo"
    random_center = random.choice(list(geo_points.values()))

    radius = 1000 * 2000 * random.random()

    return models.FieldCondition(
        key=field,
        geo_radius=models.GeoRadius(center=random_center, radius=radius),
    )


def values_count_field_condition() -> models.FieldCondition:
    field = "id_str"
    lt = random.randint(1, 5)
    gt = random.randint(2, 7)

    if lt > gt:
        if lt > 3:
            lt = None
        else:
            gt = None

    return models.FieldCondition(
        key=field,
        values_count=models.ValuesCount(lt=lt, gt=gt),
    )


def one_random_condition_please() -> models.Condition:
    return random.choice(
        [
            is_empty_condition,
            is_null_condition,
            has_id_condition,
            match_value_field_condition,
            match_text_field_condition,
            match_any_field_condition,
            match_except_field_condition,
            range_field_condition,
            datetime_range_field_condition,
            range_nested_array_field_condition,
            geo_bounding_box_field_condition,
            geo_radius_field_condition,
            values_count_field_condition,
            one_random_filter_please,
            nested_field_condition_1,
            nested_field_condition_2,
        ]
    )()


def one_random_filter_please() -> models.Filter:
    return random.choice(
        [
            must_filter,
            should_filter,
            must_not_filter,
            two_must_filter,
            two_should_filter,
            two_must_not_filter,
            should_must_filter,
            min_should_filter,
        ]
    )()


def must_filter() -> models.Filter:
    return models.Filter(must=[one_random_condition_please()])


def should_filter() -> models.Filter:
    return models.Filter(should=[one_random_condition_please()])


def min_should_filter() -> models.Filter:
    min_count = random.randint(1, 3)
    upper_bound = max(min_count + 1, min_count * 2)
    num_conditions = random.randint(min_count, upper_bound)
    return models.Filter(
        min_should=models.MinShould(
            conditions=[one_random_condition_please() for _ in range(num_conditions)],
            min_count=min_count,
        )
    )


def must_not_filter() -> models.Filter:
    return models.Filter(must_not=[one_random_condition_please()])


def two_must_filter() -> models.Filter:
    return models.Filter(must=[one_random_condition_please(), one_random_condition_please()])


def two_should_filter() -> models.Filter:
    return models.Filter(should=[one_random_condition_please(), one_random_condition_please()])


def two_must_not_filter() -> models.Filter:
    return models.Filter(must_not=[one_random_condition_please(), one_random_condition_please()])


def should_must_filter() -> models.Filter:
    return models.Filter(
        should=[one_random_condition_please()], must=[one_random_condition_please()]
    )
