import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session(max_retries=5, backoff_factor=1):
    """
    Optimized session creator with exponential backoff.
    Ensures connection drops during batch_update do not cause indefinite hangs.
    """
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        max_retries=retries, 
        pool_connections=100, 
        pool_maxsize=100
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
