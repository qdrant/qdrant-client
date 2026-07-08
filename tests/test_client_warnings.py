import warnings

import pytest

from qdrant_client.common import client_warnings


@pytest.fixture(autouse=True)
def clear_seen_messages():
    # show_warning_once dedupes via a module-level set; isolate each test.
    saved = set(client_warnings.SEEN_MESSAGES)
    client_warnings.SEEN_MESSAGES.clear()
    try:
        yield
    finally:
        client_warnings.SEEN_MESSAGES.clear()
        client_warnings.SEEN_MESSAGES.update(saved)


def _capture(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func(*args, **kwargs)
    return caught


def test_show_warning_emits_user_warning_by_default():
    caught = _capture(client_warnings.show_warning, "hello")
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert str(caught[0].message) == "hello"


def test_show_warning_honors_category():
    caught = _capture(client_warnings.show_warning, "x", DeprecationWarning)
    assert issubclass(caught[0].category, DeprecationWarning)


def test_show_warning_once_only_warns_first_time():
    first = _capture(client_warnings.show_warning_once, "only-once")
    second = _capture(client_warnings.show_warning_once, "only-once")
    assert len(first) == 1
    assert second == []


def test_show_warning_once_distinguishes_messages():
    first = _capture(client_warnings.show_warning_once, "message-a")
    other = _capture(client_warnings.show_warning_once, "message-b")
    assert len(first) == 1
    assert len(other) == 1


def test_show_warning_once_uses_idx_as_dedup_key():
    # Same idx dedupes even when the messages differ.
    first = _capture(client_warnings.show_warning_once, "text one", idx="key")
    second = _capture(client_warnings.show_warning_once, "text two", idx="key")
    assert len(first) == 1
    assert second == []
