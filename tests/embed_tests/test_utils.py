import base64
from pathlib import Path

import pytest

from qdrant_client.embed.utils import convert_paths, read_base64
from tests.utils import TESTS_PATH

EMBED_TESTS_DATA = TESTS_PATH / "embed_tests" / "misc"


def test_image_path_to_b64():
    # Test with a valid image file
    image_path = Path(EMBED_TESTS_DATA / "image.jpeg")
    original_bytes = image_path.read_bytes()

    b64_string = read_base64(image_path)
    assert isinstance(b64_string, str)

    decoded_bytes = base64.b64decode(b64_string)
    assert decoded_bytes == original_bytes, "Decoded bytes do not match original bytes"

    # Test with a non-existent file
    non_existent_path = Path(EMBED_TESTS_DATA / "gibberish.jpg")
    with pytest.raises(FileNotFoundError):
        read_base64(non_existent_path)


def test_convert_paths_preserves_prefix_endpoint():
    paths = convert_paths(["a.b", "a.b.c"])

    assert [path for root in paths for path in root.as_str_list()] == ["a.b", "a.b.c"]


def test_convert_paths_preserves_prefix_endpoint_with_siblings():
    paths = convert_paths(["a.b", "a.b.c", "a.d"])

    assert [path for root in paths for path in root.as_str_list()] == [
        "a.b",
        "a.b.c",
        "a.d",
    ]
