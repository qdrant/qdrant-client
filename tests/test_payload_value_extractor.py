import uuid

from qdrant_client.local.payload_value_extractor import parse_uuid, value_by_key


class TestValueByKey:
    def test_simple_key(self):
        assert value_by_key({"name": "x"}, "name") == ["x"]

    def test_nested_key(self):
        assert value_by_key({"address": {"city": "NYC"}}, "address.city") == ["NYC"]

    def test_list_is_flattened_by_default(self):
        assert value_by_key({"tags": [1, 2]}, "tags") == [1, 2]

    def test_list_is_kept_when_not_flat(self):
        assert value_by_key({"tags": [1, 2]}, "tags", flat=False) == [[1, 2]]

    def test_wildcard_index(self):
        payload = {"loc": [{"n": "a"}, {"n": "b"}]}
        assert value_by_key(payload, "loc[].n") == ["a", "b"]

    def test_explicit_index(self):
        payload = {"loc": [{"n": "a"}, {"n": "b"}]}
        assert value_by_key(payload, "loc[0].n") == ["a"]

    def test_missing_key_returns_none(self):
        assert value_by_key({"name": "x"}, "nope") is None

    def test_out_of_range_index_returns_none(self):
        assert value_by_key({"loc": [{"n": "a"}]}, "loc[5].n") is None


class TestParseUuid:
    def test_valid_uuid_string(self):
        text = "12345678-1234-5678-1234-567812345678"
        assert parse_uuid(text) == uuid.UUID(text)

    def test_uuid_object(self):
        value = uuid.uuid4()
        assert parse_uuid(value) == value

    def test_invalid_value_returns_none(self):
        assert parse_uuid("not-a-uuid") is None
