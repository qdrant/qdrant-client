"""SipHash-2-4 over canonical point id bytes.

Used by `SliceCondition` to split the id space into deterministic slices. The hash must match
the engine byte-for-byte, otherwise local mode would assign points to different slices than the
server does. Verified against the SipHash reference vectors and against a live Qdrant instance.
"""

import uuid

from qdrant_client.http import models

_MASK = (1 << 64) - 1

# SipHash initialization constants ("somepseudorandomlygeneratedbytes")
_INIT_0 = 0x736F6D6570736575
_INIT_1 = 0x646F72616E646F6D
_INIT_2 = 0x6C7967656E657261
_INIT_3 = 0x7465646279746573


def _rotl(x: int, b: int) -> int:
    return ((x << b) | (x >> (64 - b))) & _MASK


def _sip_round(v0: int, v1: int, v2: int, v3: int) -> tuple[int, int, int, int]:
    v0 = (v0 + v1) & _MASK
    v1 = _rotl(v1, 13)
    v1 ^= v0
    v0 = _rotl(v0, 32)
    v2 = (v2 + v3) & _MASK
    v3 = _rotl(v3, 16)
    v3 ^= v2
    v0 = (v0 + v3) & _MASK
    v3 = _rotl(v3, 21)
    v3 ^= v0
    v2 = (v2 + v1) & _MASK
    v1 = _rotl(v1, 17)
    v1 ^= v2
    v2 = _rotl(v2, 32)
    return v0, v1, v2, v3


def siphash24(data: bytes, key0: int = 0, key1: int = 0) -> int:
    """SipHash-2-4 (2 compression rounds, 4 finalization rounds). Defaults to a zero key."""
    v0 = key0 ^ _INIT_0
    v1 = key1 ^ _INIT_1
    v2 = key0 ^ _INIT_2
    v3 = key1 ^ _INIT_3

    length = len(data)
    tail_start = length - (length % 8)

    for offset in range(0, tail_start, 8):
        block = int.from_bytes(data[offset : offset + 8], "little")
        v3 ^= block
        v0, v1, v2, v3 = _sip_round(v0, v1, v2, v3)
        v0, v1, v2, v3 = _sip_round(v0, v1, v2, v3)
        v0 ^= block

    # the last block is zero-padded, with the message length in its most significant byte
    tail = data[tail_start:] + b"\x00" * (7 - (length - tail_start))
    block = int.from_bytes(tail, "little") | ((length & 0xFF) << 56)
    v3 ^= block
    v0, v1, v2, v3 = _sip_round(v0, v1, v2, v3)
    v0, v1, v2, v3 = _sip_round(v0, v1, v2, v3)
    v0 ^= block

    v2 ^= 0xFF
    for _ in range(4):
        v0, v1, v2, v3 = _sip_round(v0, v1, v2, v3)

    return v0 ^ v1 ^ v2 ^ v3


def point_id_slice(point_id: models.ExtendedPointId, total: int) -> int:
    """Index of the slice `point_id` belongs to, out of `total` disjoint slices."""
    if isinstance(point_id, int):
        # numeric ids hash over their 8 little-endian bytes
        id_bytes = point_id.to_bytes(8, "little")
    else:
        # uuids hash over their 16 RFC 4122 bytes
        id_bytes = uuid.UUID(str(point_id)).bytes

    return siphash24(id_bytes) % total
