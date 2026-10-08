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
