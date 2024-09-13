import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Union

from qdrant_client.local import datetime_utils

random_words = [
    "cat",
    "dog",
    "mouse",
    "bird",
    "fish",
    "horse",
    "cow",
    "pig",
    "sheep",
    "goat",
    "chicken",
    "duck",
    "rabbit",
    "frog",
    "snake",
    "lizard",
    "turtle",
    "bear",
    "wolf",
    "fox",
    "monkey",
    "ape",
    "gorilla",
    "elephant",
    "rhino",
    "giraffe",
    "zebra",
    "deer",
    "camel",
    "lion",
    "tiger",
    "leopard",
    "hyena",
    "jaguar",
    "cheetah",
    "kangaroo",
    "koala",
    "panda",
    "sloth",
    "hippo",
    "whale",
    "dolphin",
    "shark",
    "octopus",
    "squid",
    "crab",
    "lobster",
    "snail",
    "ant",
    "bee",
    "butterfly",
    "dragonfly",
    "mosquito",
    "fly",
    "grasshopper",
    "spider",
    "scorpion",
    "ladybug",
]

geo_points = {
    "Moscow": {"lat": 55.755826, "lon": 37.6173},
    "London": {"lat": 51.507351, "lon": -0.127758},
    "Paris": {"lat": 48.856614, "lon": 2.352222},
    "Berlin": {"lat": 52.520008, "lon": 13.404954},
    "Rome": {"lat": 41.902782, "lon": 12.496366},
    "Madrid": {"lat": 40.416775, "lon": -3.70379},
    "Barcelona": {"lat": 41.385064, "lon": 2.173403},
    "Buenos Aires": {"lat": -34.603684, "lon": -58.381559},
    "New York": {"lat": 40.712775, "lon": -74.005973},
    "Los Angeles": {"lat": 34.052235, "lon": -118.243683},
    "San Francisco": {"lat": 37.774929, "lon": -122.419416},
    "Washington": {"lat": 38.907192, "lon": -77.036871},
    "Miami": {"lat": 25.76168, "lon": -80.19179},
    "Toronto": {"lat": 43.653226, "lon": -79.383184},
    "Sydney": {"lat": -33.86882, "lon": 151.209296},
    "Melbourne": {"lat": -37.813628, "lon": 144.963058},
    "Tokyo": {"lat": 35.689487, "lon": 139.691706},
    "Osaka": {"lat": 34.693738, "lon": 135.502165},
    "Beijing": {"lat": 39.9042, "lon": 116.407396},
    "Shanghai": {"lat": 31.230416, "lon": 121.473701},
    "Hong Kong": {"lat": 22.396428, "lon": 114.109497},
    "Bangkok": {"lat": 13.756331, "lon": 100.501765},
    "Singapore": {"lat": 1.352083, "lon": 103.819836},
    "Seoul": {"lat": 37.566535, "lon": 126.977969},
    "Kuala Lumpur": {"lat": 3.139003, "lon": 101.686855},
    "Jakarta": {"lat": -6.208763, "lon": 106.845599},
    "Dubai": {"lat": 25.204849, "lon": 55.270783},
    "Cairo": {"lat": 30.04442, "lon": 31.235712},
    "Istanbul": {"lat": 41.008238, "lon": 28.978359},
    "Milan": {"lat": 45.464204, "lon": 9.189982},
    "Amsterdam": {"lat": 52.370216, "lon": 4.895168},
    "Brussels": {"lat": 50.85034, "lon": 4.35171},
    "Helsinki": {"lat": 60.169856, "lon": 24.938379},
    "Stockholm": {"lat": 59.329323, "lon": 18.068581},
    "Copenhagen": {"lat": 55.676097, "lon": 12.568337},
    "Prague": {"lat": 50.075538, "lon": 14.4378},
    "Vienna": {"lat": 48.208174, "lon": 16.373819},
    "Budapest": {"lat": 47.497912, "lon": 19.040235},
    "Warsaw": {"lat": 52.229676, "lon": 21.012229},
    "Bucharest": {"lat": 44.426767, "lon": 26.102538},
    "Athens": {"lat": 37.98381, "lon": 23.727539},
    "Lisbon": {"lat": 38.722252, "lon": -9.139337},
    "Bogota": {"lat": 4.710989, "lon": -74.072092},
    "Mexico City": {"lat": 19.432608, "lon": -99.133208},
    "Lima": {"lat": -12.046374, "lon": -77.042793},
    "Santiago": {"lat": -33.44889, "lon": -70.669266},
    "Johannesburg": {"lat": -26.204103, "lon": 28.047305},
    "Zurich": {"lat": 47.376887, "lon": 8.541694},
    "Geneva": {"lat": 46.204391, "lon": 6.143158},
    "Frankfurt": {"lat": 50.110922, "lon": 8.682127},
    "Munich": {"lat": 48.135125, "lon": 11.581981},
    "Riga": {"lat": 56.949649, "lon": 24.105186},
    "Vilnius": {"lat": 54.687157, "lon": 25.279652},
    "Tallinn": {"lat": 59.436961, "lon": 24.753575},
    "Dublin": {"lat": 53.349805, "lon": -6.26031},
    "Belfast": {"lat": 54.597285, "lon": -5.93012},
    "Manchester": {"lat": 53.480759, "lon": -2.242631},
    "Liverpool": {"lat": 53.410631, "lon": -2.97794},
    "Birmingham": {"lat": 52.486243, "lon": -1.890401},
    "Edinburgh": {"lat": 55.953252, "lon": -3.188267},
    "Glasgow": {"lat": 55.864237, "lon": -4.251806},
    "Cardiff": {"lat": 51.481581, "lon": -3.17909},
    "Bristol": {"lat": 51.454514, "lon": -2.58791},
    "Leeds": {"lat": 53.800755, "lon": -1.549077},
    "Sheffield": {"lat": 53.381129, "lon": -1.470085},
    "Newcastle": {"lat": 54.978252, "lon": -1.61778},
    "Minsk": {"lat": 53.90454, "lon": 27.561524},
    "Saint Petersburg": {"lat": 59.938732, "lon": 30.314129},
    "Kiev": {"lat": 50.4501, "lon": 30.5234},
    "Kharkiv": {"lat": 49.980812, "lon": 36.25272},
    "Odessa": {"lat": 46.47747, "lon": 30.73262},
    "Dnipro": {"lat": 48.464717, "lon": 35.046183},
    "Zaporizhzhya": {"lat": 47.82229, "lon": 35.190319},
    "Donetsk": {"lat": 48.015883, "lon": 37.80285},
    "Lviv": {"lat": 49.839683, "lon": 24.029717},
    "Kazan": {"lat": 55.78874, "lon": 49.12214},
    "Nizhny Novgorod": {"lat": 56.326887, "lon": 44.007496},
    "Chelyabinsk": {"lat": 55.16444, "lon": 61.436843},
    "Samara": {"lat": 53.195873, "lon": 50.100193},
    "Rostov-on-Don": {"lat": 47.235713, "lon": 39.7015},
}


start_datetime = datetime(2000, 1, 1)
end_datetime = datetime(2001, 1, 31)


def random_datetime() -> Union[str, datetime]:
    random_datetime = start_datetime + timedelta(
        seconds=random.randint(0, int((end_datetime - start_datetime).total_seconds())),
        microseconds=random.randint(0, 999999),
    )

    fmt = random.choice(datetime_utils.available_formats)
    if "z" in fmt:
        random_datetime = random_datetime.replace(
            tzinfo=timezone(offset=timedelta(hours=random.randint(-12, 12)))
        )

    if random.random() < 0.15:
        return random_datetime

    dt_str = random_datetime.strftime(fmt)
    return dt_str


def random_real_word():
    return random.choice(random_words)


def random_city():
    name = random.choice(list(geo_points.keys()))
    return {"name": name, "geo": geo_points[name]}


def random_signed_int():
    number = random.randint(-10, 10)
    return number


def one_random_payload_please(idx: int) -> Dict[str, Any]:
    payload = {
        "id": idx + 100,
        "id_str": [str(random.randint(1, 30)).zfill(2) for _ in range(random.randint(0, 5))],
        "text_data": uuid.uuid4().hex,
        "rand_digit": random.randint(0, 9),
        "rand_number": round(random.random(), 5),
        "rand_signed_int": random_signed_int(),
        "rand_datetime": random_datetime(),
        "text_array": [uuid.uuid4().hex, uuid.uuid4().hex],
        "words": f"{random_real_word()} {random_real_word()}",
        "nested": {
            "id": idx + 100,
            "rand_digit": random.randint(0, 9),
            "array": [
                {
                    "nested_empty": ["hello"] if random.random() < 0.5 else None,
                    "nested_empty2": ["hello"] if random.random() < 0.5 else [],
                    "word": random_real_word(),
                    "number": random.randint(1, 10),
                }
                for _ in range(random.randint(0, 5))
            ],
        },
        "nested_array": [
            [random_signed_int() for _ in range(random.randint(0, 5))]
            for _ in range(random.randint(0, 5))
        ],
        "two_words": [random_real_word(), random_real_word()],
        "city": random_city(),
        "rand_tuple": tuple(random.randint(0, 100) for _ in range(random.randint(1, 5))),
    }

    if random.random() < 0.5:
        payload["maybe"] = random_real_word()

    if random.random() < 0.5:
        payload["maybe_null"] = random_real_word()
    else:
        if random.random() < 0.5:
            payload["maybe_null"] = None

    return payload


def random_payload(num_vectors: int):
    for i in range(num_vectors):
        yield one_random_payload_please(i)
