from typing import Optional


class CommonException(Exception):
    """Base class"""


class ResourceExhaustedResponse(CommonException):
    def __init__(self, message: Optional[str], retry_after_s: Optional[int] = None) -> None:
        self.message = message
        try:
            self.retry_after_s = int(retry_after_s) if retry_after_s else 1
        except Exception:
            self.retry_after_s = 1

    def __str__(self) -> str:
        reason_phrase_str = f"({self.message})" if self.message else "(Too Many Requests)"
        status_str = f"{reason_phrase_str}".strip()
        return f"Resource Exhausted Response: {status_str}"
