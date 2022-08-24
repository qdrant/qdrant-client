__version__ = '0.1.0'

from .qdrant_client import QdrantClient


def _in_ipython() -> bool:
    """
    Check whether we're in an ipython environment, including jupyter notebooks.
    """
    try:
        eval('__IPYTHON__')
    except NameError:
        return False
    else:  # pragma: no cover
        return True


if _in_ipython():  # pragma: no cover
    # Python asyncio design is mediocre, it is not possible to await for a future, if there is another loop running.
    # Ipython uses asyncio, which makes it impossible to run other async functions, so we need to monkey-patch it.
    # It might be dangerous to do this in production, so we are doing it for Jupyter notebooks only.
    import nest_asyncio
    nest_asyncio.apply()
