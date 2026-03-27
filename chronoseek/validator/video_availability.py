import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests


@dataclass
class VideoAvailabilityResult:
    accessible: bool
    reason: str


class VideoAvailabilityChecker:
    YOUTUBE_HOSTS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    def __init__(
        self,
        cache_path: str | None = None,
        cache_ttl_seconds: int = 86400,
        timeout: int = 20,
    ):
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.timeout = max(1, int(timeout))
        self._cache = self._load_cache()

    def check(self, url: str) -> VideoAvailabilityResult:
        cached = self._get_cached(url)
        if cached is not None:
            return cached

        if self._is_youtube_url(url):
            result = self._check_youtube(url)
        else:
            result = self._check_direct_url(url)

        self._store(url, result)
        return result

    def _is_youtube_url(self, url: str) -> bool:
        host = (urlparse(url).netloc or "").lower()
        return host in self.YOUTUBE_HOSTS

    def _check_youtube(self, url: str) -> VideoAvailabilityResult:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is required for validator-side YouTube availability checks."
            ) from exc

        options = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": self.timeout,
            "extract_flat": True,
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)

            if not info:
                return VideoAvailabilityResult(False, "youtube_unavailable")

            return VideoAvailabilityResult(True, "youtube_ok")
        except Exception as exc:
            return VideoAvailabilityResult(False, str(exc))

    def _check_direct_url(self, url: str) -> VideoAvailabilityResult:
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code >= 400 or not response.ok:
                return VideoAvailabilityResult(False, f"http_{response.status_code}")
            return VideoAvailabilityResult(True, "http_ok")
        except Exception as exc:
            return VideoAvailabilityResult(False, str(exc))

    def _load_cache(self) -> dict:
        if self.cache_path is None or not self.cache_path.exists():
            return {}

        try:
            return json.loads(self.cache_path.read_text())
        except Exception:
            return {}

    def _get_cached(self, url: str) -> VideoAvailabilityResult | None:
        entry = self._cache.get(url)
        if not entry:
            return None

        checked_at = float(entry.get("checked_at", 0))
        if self.cache_ttl_seconds and time.time() - checked_at > self.cache_ttl_seconds:
            return None

        return VideoAvailabilityResult(
            accessible=bool(entry.get("accessible", False)),
            reason=str(entry.get("reason", "cached")),
        )

    def _store(self, url: str, result: VideoAvailabilityResult) -> None:
        self._cache[url] = {
            "accessible": result.accessible,
            "reason": result.reason,
            "checked_at": time.time(),
        }
        if self.cache_path is None:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
