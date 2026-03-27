import requests
import tempfile
import os
import bittensor as bt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse


class VideoDownloader:
    """
    Handles secure video downloading with retry logic.
    """

    YOUTUBE_HOSTS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    @classmethod
    def is_youtube_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return host in cls.YOUTUBE_HOSTS

    @staticmethod
    def _looks_like_video_file(path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    @classmethod
    def _download_with_requests(cls, url: str, timeout: int) -> str:
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

    @classmethod
    def _download_with_ytdlp(cls, url: str, timeout: int) -> str:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is required to download YouTube-hosted validator videos."
            ) from exc

        tmp_dir = tempfile.mkdtemp(prefix="chronoseek-ytdlp-")
        output_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        options = {
            "format": "mp4/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": timeout,
            "retries": 3,
        }

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)

        base, ext = os.path.splitext(downloaded_path)
        if ext.lower() != ".mp4":
            candidate = f"{base}.mp4"
            if os.path.exists(candidate):
                downloaded_path = candidate

        return downloaded_path

    @staticmethod
    def download_video(url: str, timeout: int = 60) -> str:
        """
        Download video to a temporary file.
        Returns: Path to temp file or empty string on failure.
        """
        try:
            if VideoDownloader.is_youtube_url(url):
                bt.logging.info("Detected YouTube URL. Downloading with yt-dlp.")
                downloaded_path = VideoDownloader._download_with_ytdlp(url, timeout)
            else:
                downloaded_path = VideoDownloader._download_with_requests(url, timeout)

            if not VideoDownloader._looks_like_video_file(downloaded_path):
                bt.logging.error(f"Downloaded file is missing or empty for URL: {url}")
                return ""

            return downloaded_path
        except Exception as e:
            bt.logging.error(f"Failed to download video: {e}")
            return ""
