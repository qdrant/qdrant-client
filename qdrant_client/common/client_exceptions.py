class CommonException(Exception):
    """Base class"""


class ResourceExhaustedResponse(CommonException):
    def __init__(self, message: str | None, retry_after_s: int | None = None) -> None:
        self.message = message
        self.retry_after_s = retry_after_s

    def __str__(self) -> str:
        reason_phrase_str = f"({self.message})" if self.message else "(Too Many Requests)"
        status_str = f"{reason_phrase_str}".strip()
        return f"Resource Exhausted Response: {status_str}"
