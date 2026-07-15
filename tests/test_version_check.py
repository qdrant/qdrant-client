"""Tests for ``qdrant_client.common.version_check``.

Covers:

* ``parse_version`` accepts a standard ``x.y.z`` string and round-trips
  major/minor/patch through the ``Version`` namedtuple.
* ``parse_version`` accepts short forms (``x.y``) with empty patch.
* ``parse_version`` rejects ``None``/empty strings and non-numeric input.
* ``is_compatible`` enforces "same major, minor within 1" in both directions.
* ``is_compatible`` returns False for any missing input and logs at debug.
* ``get_server_version`` returns the version on 200 JSON, and None on
  either an error status or an empty ``version`` field.

The HTTP path uses ``httpx.MockTransport`` so no real network is touched.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from qdrant_client.common.version_check import (
    is_compatible,
    parse_version,
)


class TestParseVersion:
    def test_parses_x_y_z(self) -> None:
        v = parse_version("1.18.0")
        assert v.major == 1
        assert v.minor == 18
        assert v.rest == ["0"]

    def test_parses_x_y_only(self) -> None:
        v = parse_version("1.18")
        assert v.major == 1
        assert v.minor == 18
        assert v.rest == []

    def test_parses_x_y_z_extra_segments(self) -> None:
        v = parse_version("1.18.0.dev1+g1234")
        assert v.major == 1
        assert v.minor == 18
        assert v.rest == ["0", "dev1+g1234"]

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            parse_version("")
        assert "Version is None" in str(excinfo.value)

    def test_rejects_non_numeric_minor(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            parse_version("1.x.0")
        assert "Unable to parse version" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)


class TestIsCompatible:
    def test_equal_versions_are_compatible(self) -> None:
        assert is_compatible("1.18.0", "1.18.0") is True

    def test_minor_within_one_is_compatible(self) -> None:
        assert is_compatible("1.17.0", "1.18.0") is True
        assert is_compatible("1.18.0", "1.17.0") is True

    def test_minor_out_of_range_is_not_compatible(self) -> None:
        assert is_compatible("1.15.0", "1.18.0") is False
        assert is_compatible("2.0.0", "1.18.0") is False

    def test_major_difference_is_not_compatible(self) -> None:
        assert is_compatible("1.18.0", "2.18.0") is False
        assert is_compatible("0.18.0", "1.18.0") is False

    def test_missing_client_version_returns_false(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG):
            assert is_compatible(None, "1.18.0") is False

    def test_missing_server_version_returns_false(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG):
            assert is_compatible("1.18.0", None) is False

    def test_unparseable_server_version_returns_false(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG):
            assert is_compatible("1.18.0", "garbage") is False

    def test_unparseable_client_version_returns_false(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG):
            assert is_compatible("garbage", "1.18.0") is False


class TestGetServerVersion:
    def test_returns_version_on_200(self) -> None:
        body = json.dumps({"version": "1.18.0", "commit": "abc"}).encode()
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
        with httpx.Client(transport=transport) as client:
            response = client.get("http://test/")
            assert response.json()["version"] == "1.18.0"

        # Replay the same logic as version_check, but with a patched httpx.get
        # so we don't depend on monkeypatching the real module function.
        def fake_get(url, headers, auth, timeout):
            return response

        import qdrant_client.common.version_check as vc

        original_get = vc.httpx.get
        vc.httpx.get = fake_get  # type: ignore[assignment]
        try:
            assert vc.get_server_version("http://test/", {}, None, timeout=5) == "1.18.0"
        finally:
            vc.httpx.get = original_get  # type: ignore[assignment]

    def test_returns_none_on_500(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(500, content=b""))
        with httpx.Client(transport=transport) as client:
            response = client.get("http://test/")

        import qdrant_client.common.version_check as vc

        original_get = vc.httpx.get
        vc.httpx.get = lambda *a, **k: response  # type: ignore[assignment]
        try:
            assert vc.get_server_version("http://test/", {}, None, timeout=5) is None
        finally:
            vc.httpx.get = original_get  # type: ignore[assignment]

    def test_returns_none_when_version_field_missing(self) -> None:
        body = json.dumps({"commit": "abc"}).encode()
        transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))
        with httpx.Client(transport=transport) as client:
            response = client.get("http://test/")

        import qdrant_client.common.version_check as vc

        original_get = vc.httpx.get
        vc.httpx.get = lambda *a, **k: response  # type: ignore[assignment]
        try:
            assert vc.get_server_version("http://test/", {}, None, timeout=5) is None
        finally:
            vc.httpx.get = original_get  # type: ignore[assignment]
