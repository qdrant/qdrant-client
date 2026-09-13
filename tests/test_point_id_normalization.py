import uuid

def normalize_point_id(point_id):
    if isinstance(point_id, int):
        if point_id < 0:
            raise ValueError("Point ID cannot be negative integer")
        return point_id
    if isinstance(point_id, str):
        val = uuid.UUID(point_id)
        return str(val).lower()
    raise TypeError("Invalid point ID type")

def test_integer_point_id():
    assert normalize_point_id(42) == 42

def test_uuid_point_id():
    uid_str = "550e8400-e29b-41d4-a716-446655440000"
    assert normalize_point_id(uid_str.upper()) == uid_str.lower()
