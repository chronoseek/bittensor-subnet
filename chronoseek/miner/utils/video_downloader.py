import requests
import tempfile
import os
import shutil
from dataclasses import dataclass
import bittensor as bt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse


@dataclass
class DownloadedVideo:
    path: str
    cleanup_paths: list[str]


class VideoDownloader:
    """
    Handles secure video downloading with retry logic.
    """

    EXTRACTOR_PLATFORM_HOSTS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "vimeo.com",
        "www.vimeo.com",
        "player.vimeo.com",
        "dailymotion.com",
        "www.dailymotion.com",
        "dai.ly",
        "tiktok.com",
        "www.tiktok.com",
        "vm.tiktok.com",
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
        "instagram.com",
        "www.instagram.com",
        "facebook.com",
        "www.facebook.com",
    }
    DIRECT_MEDIA_EXTENSIONS = {
        ".mp4",
        ".webm",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
    }

    @classmethod
    def is_extractor_platform_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return host in cls.EXTRACTOR_PLATFORM_HOSTS

    @classmethod
    def is_direct_media_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        path = (parsed.path or "").lower()
        return any(path.endswith(ext) for ext in cls.DIRECT_MEDIA_EXTENSIONS)

    @classmethod
    def should_prefer_ytdlp(cls, url: str) -> bool:
        if cls.is_extractor_platform_url(url):
            return True
        return not cls.is_direct_media_url(url)

    @staticmethod
    def _looks_like_video_file(path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    @classmethod
    def _download_with_requests(cls, url: str, timeout: int) -> DownloadedVideo:
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
            return DownloadedVideo(
                path=tmp_file.name,
                cleanup_paths=[tmp_file.name],
            )

    @classmethod
    def _download_with_ytdlp(cls, url: str, timeout: int) -> DownloadedVideo:
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

        return DownloadedVideo(
            path=downloaded_path,
            cleanup_paths=[tmp_dir],
        )

    @staticmethod
    def cleanup(downloaded_video: DownloadedVideo | None) -> None:
        if downloaded_video is None:
            return

        for cleanup_path in downloaded_video.cleanup_paths:
            try:
                if os.path.isdir(cleanup_path):
                    shutil.rmtree(cleanup_path, ignore_errors=True)
                elif os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
            except Exception as exc:
                bt.logging.warning(
                    f"Failed to clean up downloaded video artifact {cleanup_path}: {exc}"
                )

    @staticmethod
    def download_video(url: str, timeout: int = 60) -> DownloadedVideo | None:
        """
        Download video to a temporary file.
        Returns: DownloadedVideo metadata or None on failure.
        """
        attempts = []
        if VideoDownloader.should_prefer_ytdlp(url):
            bt.logging.info("Downloader strategy: yt-dlp first, HTTP fallback.")
            attempts = [
                ("yt-dlp", VideoDownloader._download_with_ytdlp),
                ("http", VideoDownloader._download_with_requests),
            ]
        else:
            bt.logging.info("Downloader strategy: HTTP first, yt-dlp fallback.")
            attempts = [
                ("http", VideoDownloader._download_with_requests),
                ("yt-dlp", VideoDownloader._download_with_ytdlp),
            ]

        last_error = None
        for method_name, downloader in attempts:
            downloaded_video = None
            try:
                downloaded_video = downloader(url, timeout)
                if not VideoDownloader._looks_like_video_file(downloaded_video.path):
                    raise RuntimeError("downloaded file is missing or empty")

                bt.logging.info(f"Video download succeeded via {method_name}.")
                return downloaded_video
            except Exception as e:
                last_error = e
                bt.logging.warning(
                    f"Video download attempt via {method_name} failed: {e}"
                )
                VideoDownloader.cleanup(downloaded_video)

        if last_error is not None:
            bt.logging.error(f"Failed to download video: {last_error}")
        return None
