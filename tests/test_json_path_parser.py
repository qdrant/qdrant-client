import pytest

from qdrant_client.local.json_path_parser import (
    JsonPathItemType,
    parse_json_path,
)


def as_tuples(key):
    return [(i.item_type, i.key, i.index) for i in parse_json_path(key)]


KEY = JsonPathItemType.KEY
INDEX = JsonPathItemType.INDEX
WILDCARD = JsonPathItemType.WILDCARD_INDEX


class TestParseJsonPath:
    """Unit tests for local json-path parsing.

    ``parse_json_path`` splits a payload json path into key / index /
    wildcard-index items and rejects malformed paths. It backs local-mode
    payload filtering but had no direct test coverage.
    """

    def test_single_key(self):
        assert as_tuples("a") == [(KEY, "a", None)]

    def test_dotted_keys(self):
        assert as_tuples("a.b.c") == [
            (KEY, "a", None),
            (KEY, "b", None),
            (KEY, "c", None),
        ]

    def test_key_allows_underscore_and_hyphen(self):
        assert as_tuples("a_b-c") == [(KEY, "a_b-c", None)]

    def test_index(self):
        assert as_tuples("a[0]") == [(KEY, "a", None), (INDEX, None, 0)]

    def test_multiple_indices(self):
        assert as_tuples("a[0][1]") == [
            (KEY, "a", None),
            (INDEX, None, 0),
            (INDEX, None, 1),
        ]

    def test_negative_index(self):
        assert as_tuples("a[-1]") == [(KEY, "a", None), (INDEX, None, -1)]

    def test_wildcard_index(self):
        assert as_tuples("a[]") == [(KEY, "a", None), (WILDCARD, None, None)]

    def test_wildcard_then_key(self):
        assert as_tuples("a[].b") == [
            (KEY, "a", None),
            (WILDCARD, None, None),
            (KEY, "b", None),
        ]

    def test_index_then_key(self):
        assert as_tuples("a[0][1].b") == [
            (KEY, "a", None),
            (INDEX, None, 0),
            (INDEX, None, 1),
            (KEY, "b", None),
        ]

    def test_quoted_key_preserves_special_chars(self):
        # A quoted segment is a single key even though it contains a dot.
        assert as_tuples('"a.b"') == [(KEY, "a.b", None)]

    def test_quoted_key_followed_by_key(self):
        assert as_tuples('"a".b') == [(KEY, "a", None), (KEY, "b", None)]

    @pytest.mark.parametrize(
        "path",
        [
            "",  # empty
            ".a",  # leading separator
            "a.",  # trailing separator
            "a[0",  # unclosed bracket
            "a[x]",  # non-integer index
            '"ab',  # unclosed quote
        ],
    )
    def test_invalid_paths_raise(self, path):
        with pytest.raises(ValueError):
            parse_json_path(path)
