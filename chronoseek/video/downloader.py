import requests
import tempfile
import os
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import bittensor as bt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

from chronoseek.hippius.s3 import (
    DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
    DEFAULT_HIPPIUS_S3_REGION,
    HippiusS3Config,
    HippiusS3StorageClient,
    download_public_file,
    parse_hippius_s3_url,
)


@dataclass
class DownloadedVideo:
    path: str
    cleanup_paths: list[str]


@dataclass(frozen=True)
class DownloadAttemptFailure:
    backend: str
    message: str


class VideoDownloadError(RuntimeError):
    def __init__(self, *, url: str, failures: list[DownloadAttemptFailure]):
        self.url = url
        self.failures = list(failures)
        summary = "; ".join(
            f"{failure.backend}: {failure.message}" for failure in self.failures
        )
        super().__init__(summary or "video download failed")

    def details(self) -> dict:
        return {
            "video_url": self.url,
            "download_attempts": [
                {
                    "backend": failure.backend,
                    "message": failure.message,
                }
                for failure in self.failures
            ],
        }


class VideoDownloader:
    """
    Shared validator/miner video downloader with retry logic.

    Extractor platform pages such as YouTube and Vimeo are routed through
    yt-dlp. Direct media URLs prefer plain HTTP first.
    """

    # Netscape-format cookies.txt (see yt-dlp wiki: passing cookies to yt-dlp).
    _ENV_YTDLP_COOKIES_FILE = "YTDLP_COOKIES"
    # Optional: e.g. "chrome", "firefox", or "chrome:Default" (profile). Chutes
    # runtimes usually need _ENV_YTDLP_COOKIES_FILE instead.
    _ENV_YTDLP_COOKIES_BROWSER = "YTDLP_COOKIES_BROWSER"
    # NOTE: This is intentionally opt-in. Defaulting to chrome breaks headless
    # servers/CI where no Chrome profile exists and yt-dlp will error out.
    _DEFAULT_YTDLP_COOKIES_BROWSER = ""
    # yt-dlp EJS n-challenge solver needs Node 20+ or Deno 2+ (see yt-dlp wiki/EJS).
    _ENV_YTDLP_NODE_PATH = "YTDLP_NODE_PATH"
    _ENV_YTDLP_DENO_PATH = "YTDLP_DENO_PATH"
    _ENV_HIPPIUS_S3_ENDPOINT_URL = "HIPPIUS_S3_ENDPOINT_URL"
    _ENV_HIPPIUS_S3_PUBLIC_BASE_URL = "HIPPIUS_S3_PUBLIC_BASE_URL"
    _ENV_HIPPIUS_S3_BUCKET = "HIPPIUS_S3_BUCKET"
    _ENV_HIPPIUS_S3_ACCESS_KEY_ID = "HIPPIUS_S3_ACCESS_KEY_ID"
    _ENV_HIPPIUS_S3_SECRET_ACCESS_KEY = "HIPPIUS_S3_SECRET_ACCESS_KEY"
    _ENV_HIPPIUS_S3_REGION = "HIPPIUS_S3_REGION"
    # Node-based parents (e.g. PM2) set these; yt-dlp's node/deno children then break IPC.
    _YTDLP_STRIP_ENV_KEYS = ("NODE_CHANNEL_FD", "NODE_CHANNEL_SERIALIZATION_MODE")

    HIPPIUS_HOSTS = {
        "s3.hippius.com",
    }
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
    def is_hippius_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        if host in cls.HIPPIUS_HOSTS or host.endswith(".hippius.com"):
            return True
        return host in cls._configured_hippius_hosts()

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

    @staticmethod
    def _looks_like_video_container(path: str) -> bool:
        """
        Reject HTML error pages and other non-media saved with a .mp4 name.
        MP4 ISO BMFF contains 'ftyp' in the first box; WebM uses EBML header.
        """
        try:
            with open(path, "rb") as handle:
                head = handle.read(2048)
        except OSError:
            return False
        if not head:
            return False
        stripped = head.lstrip()
        if stripped.startswith((b"<", b"<!")):
            return False
        upper = head[:512].upper()
        if b"<!DOCTYPE" in upper or b"<HTML" in upper:
            return False
        if b"ftyp" in head[:32]:
            return True
        if head.startswith(b"\x1a\x45\xdf\xa3"):
            return True
        if head.startswith(b"RIFF") and b"AVI " in head[:20]:
            return True
        return False

    @staticmethod
    def _env_value(name: str, default: str = "") -> str:
        return os.environ.get(name, "").strip() or default

    @classmethod
    def _configured_hippius_hosts(cls) -> set[str]:
        hosts: set[str] = set()
        for env_name in (
            cls._ENV_HIPPIUS_S3_ENDPOINT_URL,
            cls._ENV_HIPPIUS_S3_PUBLIC_BASE_URL,
        ):
            value = cls._env_value(env_name)
            if not value:
                continue
            parsed = urlparse(value)
            host = (parsed.netloc or parsed.path or "").split("/", 1)[0].lower()
            if host:
                hosts.add(host)
        return hosts

    @classmethod
    def _download_attempts(cls, url: str) -> list[tuple[str, Callable[..., DownloadedVideo]]]:
        """
        Order download backends. Plain GET on extractor pages often returns HTML;
        the HTTP backend still validates Content-Type and container bytes.
        """
        if cls.is_hippius_url(url):
            return [
                ("hippius-s3", cls._download_with_hippius),
                ("http", cls._download_with_requests),
            ]

        if cls.should_prefer_ytdlp(url):
            return [
                ("yt-dlp", cls._download_with_ytdlp),
                ("http", cls._download_with_requests),
            ]

        return [
            ("http", cls._download_with_requests),
            ("yt-dlp", cls._download_with_ytdlp),
        ]

    @classmethod
    def _download_with_hippius(cls, url: str, timeout: int) -> DownloadedVideo:
        """
        Download a public Hippius S3 task artifact.

        Validator-generated tasks are published as public S3-compatible objects.
        Miners should route those URLs through this branch first so synthetic
        Hippius artifacts do not get sent through extractor heuristics.
        """
        public_base_url = cls._env_value(
            cls._ENV_HIPPIUS_S3_PUBLIC_BASE_URL,
            DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
        )
        configured_bucket = cls._env_value(cls._ENV_HIPPIUS_S3_BUCKET)
        object_ref = parse_hippius_s3_url(
            url,
            default_bucket=configured_bucket,
            public_base_url=public_base_url,
        )
        access_key_id = cls._env_value(cls._ENV_HIPPIUS_S3_ACCESS_KEY_ID)
        secret_access_key = cls._env_value(cls._ENV_HIPPIUS_S3_SECRET_ACCESS_KEY)
        has_credentials = bool(access_key_id and secret_access_key)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            local_path = tmp_file.name

        try:
            if object_ref:
                client = HippiusS3StorageClient(
                    HippiusS3Config(
                        endpoint_url=cls._env_value(
                            cls._ENV_HIPPIUS_S3_ENDPOINT_URL,
                            DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
                        ),
                        public_base_url=public_base_url,
                        bucket=object_ref.bucket,
                        access_key_id=access_key_id,
                        secret_access_key=secret_access_key,
                        region=cls._env_value(
                            cls._ENV_HIPPIUS_S3_REGION,
                            DEFAULT_HIPPIUS_S3_REGION,
                        ),
                        anonymous=not has_credentials,
                    ),
                    timeout_seconds=timeout,
                )
                client.download_file(
                    object_key=object_ref.object_key,
                    local_path=local_path,
                )
            else:
                download_public_file(
                    url=url,
                    local_path=local_path,
                    timeout_seconds=timeout,
                )
        except Exception:
            try:
                os.remove(local_path)
            except OSError:
                pass
            raise

        return DownloadedVideo(
            path=local_path,
            cleanup_paths=[local_path],
        )

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

        ctype = (response.headers.get("Content-Type") or "").lower().split(";", 1)[0].strip()
        if ctype.startswith("text/") or ctype in ("application/json", "application/javascript"):
            raise RuntimeError(f"URL returned non-media Content-Type: {ctype or 'missing'}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            return DownloadedVideo(
                path=tmp_file.name,
                cleanup_paths=[tmp_file.name],
            )

    @classmethod
    def _ytdlp_cookie_options(cls) -> dict:
        """Options for YouTube auth when Google serves the bot / sign-in interstitial."""
        opts: dict = {}
        path = cls._env_value(cls._ENV_YTDLP_COOKIES_FILE)
        if path:
            path = os.path.expanduser(path)
            if os.path.isfile(path):
                opts["cookiefile"] = path
            else:
                bt.logging.warning(
                    f"{cls._ENV_YTDLP_COOKIES_FILE} is set but not a readable file: {path}"
                )
        browser = cls._env_value(
            cls._ENV_YTDLP_COOKIES_BROWSER,
            cls._DEFAULT_YTDLP_COOKIES_BROWSER,
        )
        if browser and "cookiefile" not in opts:
            parts = tuple(p for p in browser.split(":") if p)
            if parts:
                opts["cookiesfrombrowser"] = parts
        return opts

    @staticmethod
    @contextmanager
    def _yt_dlp_clean_parent_env():
        keys = VideoDownloader._YTDLP_STRIP_ENV_KEYS
        saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
        try:
            yield
        finally:
            os.environ.update(saved)

    @classmethod
    def _ytdlp_js_runtime_options(cls) -> dict:
        """Enable YouTube n/sig challenge solving (requires yt-dlp-ejs + supported runtime)."""
        runtimes: dict[str, dict[str, str]] = {}
        node_path = cls._env_value(cls._ENV_YTDLP_NODE_PATH)
        runtimes["node"] = (
            {"path": os.path.expanduser(node_path)} if node_path else {}
        )
        deno_path = cls._env_value(cls._ENV_YTDLP_DENO_PATH)
        runtimes["deno"] = (
            {"path": os.path.expanduser(deno_path)} if deno_path else {}
        )
        return {"js_runtimes": runtimes}

    @staticmethod
    def _is_youtube_bot_or_signin_error(message: str) -> bool:
        lower = message.lower()
        return (
            "confirm you’re not a bot" in lower
            or "confirm you're not a bot" in lower
            or "sign in to confirm" in lower
        )

    @classmethod
    def _download_with_ytdlp(cls, url: str, timeout: int) -> DownloadedVideo:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is required to download extractor-platform videos."
            ) from exc

        tmp_dir = tempfile.mkdtemp(prefix="chronoseek-ytdlp-")
        output_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        cookie_opts = cls._ytdlp_cookie_options()
        js_opts = cls._ytdlp_js_runtime_options()
        options = {
            "format": "mp4/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": timeout,
            "retries": 3,
            # Prefer mobile/web clients first; many videos work without cookies; bot-checked
            # IDs still need YTDLP_COOKIES or YTDLP_COOKIES_BROWSER.
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web", "ios"],
                },
            },
            **js_opts,
            **cookie_opts,
        }
        downloaded_path = ""
        try:
            with cls._yt_dlp_clean_parent_env():
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(url, download=True)
                    downloaded_path = ydl.prepare_filename(info)
        except Exception as exc:
            err_text = str(exc)
            if cls._is_youtube_bot_or_signin_error(err_text):
                if not cookie_opts:
                    bt.logging.error(
                        "YouTube blocked this download (bot check). Export cookies from a "
                        "logged-in browser and set "
                        f"{cls._ENV_YTDLP_COOKIES_FILE} to the cookies.txt path, or set "
                        f"{cls._ENV_YTDLP_COOKIES_BROWSER} explicitly, for example "
                        "chrome:Default on a machine with a readable Chrome profile. See "
                        "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
                    )
                else:
                    bt.logging.error(
                        "YouTube still blocked this download after applying cookies "
                        f"({cls._ENV_YTDLP_COOKIES_FILE} / "
                        f"{cls._ENV_YTDLP_COOKIES_BROWSER}). Re-export a fresh cookies.txt "
                        "from a browser session that can play this video, ensure the runtime "
                        "process inherits that env var, and confirm the file path is readable."
                    )
            raise

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
    def download_video(
        url: str,
        timeout: int = 60,
        *,
        raise_on_failure: bool = False,
    ) -> DownloadedVideo | None:
        """
        Download video to a temporary file.
        Returns: DownloadedVideo metadata or None on failure.
        """
        attempts = VideoDownloader._download_attempts(url)
        if len(attempts) == 1:
            bt.logging.info(f"Downloader strategy: {attempts[0][0]} only.")
        elif attempts[0][0] == "hippius-s3":
            bt.logging.info("Downloader strategy: Hippius S3 first, HTTP fallback.")
        elif attempts[0][0] == "yt-dlp":
            bt.logging.info("Downloader strategy: yt-dlp first, HTTP fallback.")
        else:
            bt.logging.info("Downloader strategy: HTTP first, yt-dlp fallback.")

        failures: list[DownloadAttemptFailure] = []
        for method_name, downloader in attempts:
            downloaded_video = None
            try:
                downloaded_video = downloader(url, timeout)
                if not VideoDownloader._looks_like_video_file(downloaded_video.path):
                    raise RuntimeError("downloaded file is missing or empty")
                if not VideoDownloader._looks_like_video_container(downloaded_video.path):
                    raise RuntimeError(
                        "downloaded file is not a video container (often HTML from a watch page)"
                    )

                bt.logging.info(f"Video download succeeded via {method_name}.")
                return downloaded_video
            except Exception as e:
                failures.append(
                    DownloadAttemptFailure(
                        backend=method_name,
                        message=str(e),
                    )
                )
                bt.logging.warning(
                    f"Video download attempt via {method_name} failed: {e}"
                )
                VideoDownloader.cleanup(downloaded_video)

        if failures:
            error = VideoDownloadError(url=url, failures=failures)
            bt.logging.error(f"Failed to download video: {error}")
            if raise_on_failure:
                raise error
        return None
