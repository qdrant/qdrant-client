import random
import uuid

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
}


def random_real_word():
    return random.choice(random_words)


def random_city():
    name = random.choice(list(geo_points.keys()))
    return {"name": name, "geo": geo_points[name]}


def one_random_payload_please(idx):
    return {
        "id": idx + 100,
        "id_str": [str(random.randint(1, 30)).zfill(2) for _ in range(random.randint(0, 5))],
        "text_data": uuid.uuid4().hex,
        "rand_number": round(random.random(), 5),
        "text_array": [uuid.uuid4().hex, uuid.uuid4().hex],
        "words": f"{random_real_word()} {random_real_word()}",
        "nested": {
            "id": idx + 100,
            "array": [
                {
                    "word": random_real_word(),
                    "number": random.randint(1, 10),
                }
                for _ in range(random.randint(0, 5))
            ],
        },
        "city": random_city(),
    }


def random_payload(num_vectors):
    for i in range(num_vectors):
        yield one_random_payload_please(i)
