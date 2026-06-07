from qdrant_client.http.api_client import parse_as_type
from qdrant_client.http.models import models as m


def test_flat_hardware_usage_response_is_parsed_as_hardware_usage():
    response = {
        "usage": {
            "cpu": 0,
            "payload_io_read": 1,
            "payload_io_write": 2,
            "payload_index_io_read": 3,
            "payload_index_io_write": 4,
            "vector_io_read": 5,
            "vector_io_write": 6,
        },
        "time": 0.1,
        "status": "ok",
        "result": None,
    }

    parsed = parse_as_type(response, m.InlineResponse2006)

    assert parsed.usage is not None
    assert parsed.usage.hardware == m.HardwareUsage(
        cpu=0,
        payload_io_read=1,
        payload_io_write=2,
        payload_index_io_read=3,
        payload_index_io_write=4,
        vector_io_read=5,
        vector_io_write=6,
    )


def test_nested_hardware_usage_response_is_preserved():
    response = {
        "usage": {
            "hardware": {
                "cpu": 0,
                "payload_io_read": 1,
                "payload_io_write": 2,
                "payload_index_io_read": 3,
                "payload_index_io_write": 4,
                "vector_io_read": 5,
                "vector_io_write": 6,
            }
        },
        "time": 0.1,
        "status": "ok",
        "result": None,
    }

    parsed = parse_as_type(response, m.InlineResponse2006)

    assert parsed.usage is not None
    assert parsed.usage.hardware == m.HardwareUsage(
        cpu=0,
        payload_io_read=1,
        payload_io_write=2,
        payload_index_io_read=3,
        payload_index_io_write=4,
        vector_io_read=5,
        vector_io_write=6,
    )
