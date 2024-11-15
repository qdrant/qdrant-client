import warnings
from typing import Optional

SEEN_DEPRECATIONS = set()


def deprecation_warning(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def deprecation_warning_once(message: str, idx: Optional[str] = None) -> None:
    """
    Same as deprecation_warning, but will be shown only once per program run.
    """

    key = idx if idx is not None else message

    if key not in SEEN_DEPRECATIONS:
        SEEN_DEPRECATIONS.add(key)
    else:
        return

    warnings.warn(message, DeprecationWarning, stacklevel=2)
