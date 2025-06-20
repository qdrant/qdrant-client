import base64
from pathlib import Path

import pytest

from qdrant_client.embed.utils import to_base64
from tests.utils import TESTS_PATH

EMBED_TESTS_DATA = TESTS_PATH / "embed_tests" / "misc"


def test_image_path_to_b64():
    # Test with a valid image file
    image_path = Path(EMBED_TESTS_DATA / "image.jpeg")
    original_bytes = image_path.read_bytes()

    b64_string = to_base64(image_path)
    assert isinstance(b64_string, str)

    decoded_bytes = base64.b64decode(b64_string)
    assert decoded_bytes == original_bytes, "Decoded bytes do not match original bytes"

    # Test with a non-existent file
    non_existent_path = Path(EMBED_TESTS_DATA / "gibberish.jpg")
    with pytest.raises(FileNotFoundError):
        to_base64(non_existent_path)
    try:
        to_base64(non_existent_path)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
