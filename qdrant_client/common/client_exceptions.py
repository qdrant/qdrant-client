class QdrantException(Exception):
    """Base class"""


class ResourceExhaustedResponse(QdrantException):
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


class ResourceQuotaExceeded(QdrantException):
    def __init__(self, message: str) -> None:
        self.message = message if message else "Quota Exceeded Response"

    def __str__(self) -> str:
        return self.message.strip()
