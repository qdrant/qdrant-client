class QdrantException(Exception):
    """Base class"""


class ResourceExhaustedResponse(QdrantException):
    def __init__(self, message: str, retry_after_s: int) -> None:
        self.message = message
        try:
            self.retry_after_s = int(retry_after_s) if retry_after_s else 1
        except Exception:
            self.retry_after_s = 1

    def __str__(self) -> str:
        reason_phrase_str = f"{self.message}" if self.message else "Resource Exhausted Response"
        return f"{reason_phrase_str}".strip()
