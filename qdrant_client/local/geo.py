from math import asin, cos, radians, sin, sqrt


def geo_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate distance between two points on Earth using Haversine formula.

    Args:
        lon1: longitude of first point
        lat1: latitude of first point
        lon2: longitude of second point
        lat2: latitude of second point

    Returns:
        distance in meters
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km * 1000


def test_geo_distance() -> None:
    moscow = {"lon": 37.6173, "lat": 55.7558}
    london = {"lon": -0.1278, "lat": 51.5074}
    berlin = {"lon": 13.4050, "lat": 52.5200}

    assert geo_distance(moscow["lon"], moscow["lat"], moscow["lon"], moscow["lat"]) < 1.0

    assert geo_distance(moscow["lon"], moscow["lat"], london["lon"], london["lat"]) > 2400 * 1000
    assert geo_distance(moscow["lon"], moscow["lat"], london["lon"], london["lat"]) < 2600 * 1000
    assert geo_distance(moscow["lon"], moscow["lat"], berlin["lon"], berlin["lat"]) > 1600 * 1000
    assert geo_distance(moscow["lon"], moscow["lat"], berlin["lon"], berlin["lat"]) < 1650 * 1000
