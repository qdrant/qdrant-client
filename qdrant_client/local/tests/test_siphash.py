import uuid

import pytest

from qdrant_client.local.siphash import point_id_slice, siphash24

# Reference vectors from the SipHash paper (Aumasson & Bernstein), for the key
# 000102...0f and messages 00, 0001, 000102, ... Only the first few are needed to pin
# the implementation down; a single wrong rotation or constant breaks all of them.
REFERENCE_KEY = bytes(range(16))
REFERENCE_VECTORS = [
    0x726FDB47DD0E0E31,
    0x74F839C593DC67FD,
    0x0D6C8009D9A94F5A,
    0x85676696D7FB7E2D,
    0xCF2794E0277187B7,
    0x18765564CD99A68D,
    0xCBC9466E58FEE3CE,
    0xAB0200F58B01D137,
]


def test_siphash24_reference_vectors():
    key0 = int.from_bytes(REFERENCE_KEY[:8], "little")
    key1 = int.from_bytes(REFERENCE_KEY[8:], "little")

    for length, expected in enumerate(REFERENCE_VECTORS):
        message = bytes(range(length))
        assert siphash24(message, key0, key1) == expected, f"mismatch for {length}-byte message"


def test_siphash24_defaults_to_zero_key():
    assert siphash24(b"") == siphash24(b"", 0, 0)


def test_siphash24_stays_in_u64_range():
    for length in range(0, 40):
        assert 0 <= siphash24(bytes(range(length))) < 2**64


@pytest.mark.parametrize(
    "point_id",
    [0, 1, 2**32, 2**53 + 1, 2**63, 2**64 - 1, "00000000-0000-0000-0000-000000000000"],
)
def test_point_id_slice_is_in_range(point_id):
    for total in (1, 2, 7, 64):
        assert 0 <= point_id_slice(point_id, total) < total


def test_point_id_slice_total_one_takes_everything():
    for point_id in [0, 17, 2**64 - 1, str(uuid.uuid4())]:
        assert point_id_slice(point_id, 1) == 0


def test_point_id_slice_uuid_accepts_both_str_and_uuid():
    point_id = uuid.uuid4()
    assert point_id_slice(str(point_id), 8) == point_id_slice(point_id, 8)


def test_point_id_slice_uuid_uses_rfc4122_byte_order():
    # `bytes_le` reorders the first 8 bytes, which would silently disagree with the engine.
    # These uuids were chosen so the two encodings land in different slices.
    point_id = uuid.UUID("00112233-4455-6677-8899-aabbccddeeff")
    assert point_id_slice(point_id, 4) == siphash24(point_id.bytes) % 4
    assert siphash24(point_id.bytes) % 4 != siphash24(point_id.bytes_le) % 4


def test_point_id_slice_numeric_uses_little_endian():
    assert point_id_slice(1, 4) == siphash24((1).to_bytes(8, "little")) % 4


def test_point_id_slice_is_nested_across_totals():
    """Slices with different totals share the same hash, so a finer slice sits inside a coarser one.

    Documented property: slice 0 of total=4 is a strict subset of slice 0 of total=2.
    """
    ids = list(range(500))
    fine = {i for i in ids if point_id_slice(i, 4) == 0}
    coarse = {i for i in ids if point_id_slice(i, 2) == 0}
    assert fine and fine < coarse


def test_point_id_slice_partitions_the_id_space():
    ids = list(range(1000)) + [str(uuid.UUID(int=i)) for i in range(200)]
    for total in (1, 3, 8, 32):
        buckets = [set() for _ in range(total)]
        for point_id in ids:
            buckets[point_id_slice(point_id, total)].add(str(point_id))

        assigned = [x for bucket in buckets for x in bucket]
        assert len(assigned) == len(ids), f"total={total} did not cover every id exactly once"
        assert len(set(assigned)) == len(ids), f"total={total} produced overlapping slices"


def test_point_id_slice_is_roughly_uniform():
    ids = list(range(10_000))
    total = 8
    counts = [0] * total
    for point_id in ids:
        counts[point_id_slice(point_id, total)] += 1

    expected = len(ids) / total
    # a keyed hash should stay well inside 20% of the mean for this sample size
    assert all(abs(count - expected) < expected * 0.2 for count in counts), counts
