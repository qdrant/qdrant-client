import warnings
from typing import Optional

SEEN_WARNINGS = set()


def user_warning(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def user_warning_once(message: str, idx: Optional[str] = None) -> None:
    """
    Same as user_warning, but will be shown only once per program run.
    """

    key = idx if idx is not None else message

    if key not in SEEN_WARNINGS:
        SEEN_WARNINGS.add(key)
    else:
        return

    user_warning(message=message)
