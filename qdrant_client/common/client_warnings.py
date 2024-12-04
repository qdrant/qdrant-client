import warnings
from typing import Optional

SEEN_MESSAGES = set()


def show_warning(message: str, category: type[Warning] = UserWarning) -> None:
    warnings.warn(message, category, stacklevel=4)


def show_warning_once(
    message: str, category: type[Warning] = UserWarning, idx: Optional[str] = None
) -> None:
    """
    Show a warning of the specified category only once per program run.
    """
    key = idx if idx is not None else message

    if key not in SEEN_MESSAGES:
        SEEN_MESSAGES.add(key)
        show_warning(message, category)
