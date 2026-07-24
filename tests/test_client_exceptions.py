"""Tests for ``qdrant_client.common.client_exceptions``.

Pure unit tests. No network, no fixtures.

Covers:

* ``QdrantException`` is a regular ``Exception`` subclass (catchable as either).
* ``ResourceExhaustedResponse`` stores ``retry_after_s`` as an int and
  falls back to a default message when given an empty string.
* ``ResourceExhaustedResponse.__str__`` strips surrounding whitespace.
* ``ResourceExhaustedResponse`` raises ``QdrantException`` (chained) when
  given a non-integer ``retry_after_s``.
"""

from __future__ import annotations

import pytest

from qdrant_client.common.client_exceptions import (
    QdrantException,
    ResourceExhaustedResponse,
)


def test_qdrant_exception_is_exception_subclass() -> None:
    assert issubclass(QdrantException, Exception)
    err = QdrantException("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"


def test_resource_exhausted_response_stores_int_retry_after() -> None:
    err = ResourceExhaustedResponse("slow down", retry_after_s="5")
    assert err.retry_after_s == 5
    assert isinstance(err.retry_after_s, int)


def test_resource_exhausted_response_uses_default_message_when_empty() -> None:
    err = ResourceExhaustedResponse("", retry_after_s=10)
    assert str(err) == "Resource Exhausted Response"


def test_resource_exhausted_response_str_strips_whitespace() -> None:
    err = ResourceExhaustedResponse("  rate limited  \n", retry_after_s=2)
    assert str(err) == "rate limited"


def test_resource_exhausted_response_keeps_provided_message() -> None:
    err = ResourceExhaustedResponse("custom message", retry_after_s=1)
    assert err.message == "custom message"
    assert str(err) == "custom message"


def test_resource_exhausted_response_rejects_non_integer_retry_after() -> None:
    with pytest.raises(QdrantException) as excinfo:
        ResourceExhaustedResponse("slow down", retry_after_s="not-an-int")
    assert "Retry-After header value is not a valid integer" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_resource_exhausted_response_rejects_float_string_retry_after() -> None:
    # "1.5" parses as float, not int; int("1.5") raises ValueError.
    with pytest.raises(QdrantException):
        ResourceExhaustedResponse("slow down", retry_after_s="1.5")


def test_resource_exhausted_response_is_qdrant_exception() -> None:
    err = ResourceExhaustedResponse("slow down", retry_after_s=5)
    assert isinstance(err, QdrantException)
    assert isinstance(err, Exception)
