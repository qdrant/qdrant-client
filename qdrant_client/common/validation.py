from collections.abc import Sequence
from itertools import islice
from typing import Any, Iterable, Iterator

import numpy as np

from qdrant_client.http import models


def validate_batch_lengths(batch: models.Batch) -> None:
    """Reject a batch whose vector or payload columns are not as long as its ids.

    The server validates this for REST only. gRPC has no batch shape, so the client splits
    the batch into points itself, and indexing the columns by position would silently drop
    extra values.
    """
    num_ids = len(batch.ids)
    columns = batch.vectors.values() if isinstance(batch.vectors, dict) else [batch.vectors]
    for vectors in columns:
        if len(vectors) != num_ids:
            raise ValueError(
                f"number of ids and vectors must be equal ({num_ids} != {len(vectors)})"
            )
    if batch.payloads is not None and len(batch.payloads) != num_ids:
        raise ValueError(
            f"number of ids and payloads must be equal ({num_ids} != {len(batch.payloads)})"
        )


def vectors_length(vectors: Any) -> int | None:
    """How many points `upload_collection` uploads, None for generators and other streams.

    Only lists, tuples and arrays count: other objects with `__len__`, e.g. `tqdm(generator)`,
    may not know their length.
    """
    if isinstance(vectors, dict):  # numpy arrays of named vectors, all as long
        return len(next(iter(vectors.values()))) if vectors else 0
    return len(vectors) if isinstance(vectors, (Sequence, np.ndarray)) else None


def require_length(items: Iterable | None, size: int, name: str) -> Iterable | None:
    """Makes `ids` or `payload` of `upload_collection` raise if they end before `size` vectors.

    Checked at once for lists, tuples and arrays, otherwise when the stream runs out. They may be
    longer, e.g. `itertools.count()`: a stream is read only up to `size` items.
    """
    if items is None:
        return None
    if isinstance(items, (Sequence, np.ndarray)):
        if len(items) < size:
            raise ValueError(f"{name} iterable is shorter than vectors iterable")
        return items
    return _take_exactly(items, size, name)


def _take_exactly(items: Iterable, size: int, name: str) -> Iterator:
    taken = 0
    for item in islice(items, size):
        yield item
        taken += 1
    if taken < size:
        raise ValueError(f"{name} iterable is shorter than vectors iterable")
