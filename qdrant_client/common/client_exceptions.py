from grpc.aio import AioRpcError


class QdrantException(Exception):
    """Base class"""


class ResourceExhaustedResponse(QdrantException, AioRpcError):
    def __init__(self, message: str, retry_after_s: int) -> None:
        self.message = message if message else "Resource Exhausted Response"
        try:
            self.retry_after_s = int(retry_after_s)
        except Exception:
            raise QdrantException(
                f"Retry-After header value is not a valid integer: {retry_after_s}"
            )

    def __str__(self) -> str:
        return self.message.strip()
