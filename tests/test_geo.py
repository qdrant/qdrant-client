from qdrant_client.local.geo import boolean_point_in_polygon, geo_distance

# A 4x4 axis-aligned square, and a 2x2 square hole in its center.
SQUARE = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0), (0.0, 0.0)]
HOLE = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0), (1.0, 1.0)]


class TestGeoDistance:
    """Haversine great-circle distance used for local-mode geo scoring."""

    def test_zero_for_same_point(self):
        assert geo_distance(37.6173, 55.7558, 37.6173, 55.7558) == 0.0

    def test_known_city_distance(self):
        # Moscow <-> London is roughly 2500 km.
        meters = geo_distance(37.6173, 55.7558, -0.1278, 51.5074)
        assert 2_400_000 < meters < 2_600_000

    def test_symmetric(self):
        forward = geo_distance(37.6173, 55.7558, -0.1278, 51.5074)
        backward = geo_distance(-0.1278, 51.5074, 37.6173, 55.7558)
        assert forward == backward


class TestPointInPolygon:
    """Local-mode geo-polygon filtering (ray casting with hole support)."""

    def test_point_inside(self):
        assert boolean_point_in_polygon((2.0, 2.0), SQUARE, []) is True

    def test_point_outside(self):
        assert boolean_point_in_polygon((5.0, 5.0), SQUARE, []) is False

    def test_point_in_hole_is_outside(self):
        # Inside the exterior ring but inside a hole -> not in the polygon.
        assert boolean_point_in_polygon((2.0, 2.0), SQUARE, [HOLE]) is False

    def test_point_inside_but_outside_hole(self):
        assert boolean_point_in_polygon((0.5, 0.5), SQUARE, [HOLE]) is True

    def test_point_on_exterior_boundary_is_outside(self):
        # boolean_point_in_polygon checks the exterior with ignore_boundary=True.
        assert boolean_point_in_polygon((0.0, 2.0), SQUARE, []) is False
