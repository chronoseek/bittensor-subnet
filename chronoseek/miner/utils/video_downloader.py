import requests
import tempfile
import bittensor as bt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class VideoDownloader:
    """
    Handles secure video downloading with retry logic.
    """

    @staticmethod
    def download_video(url: str, timeout: int = 60) -> str:
        """
        Download video to a temporary file.
        Returns: Path to temp file or empty string on failure.
        """
        try:
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            http = requests.Session()
            http.mount("https://", adapter)
            http.mount("http://", adapter)

            response = http.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                return tmp_file.name
        except Exception as e:
            bt.logging.error(f"Failed to download video: {e}")
            return ""
