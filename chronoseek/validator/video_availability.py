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


class _SilentYtDlpLogger:
    def debug(self, msg):
        return None

    def warning(self, msg):
        return None

    def error(self, msg):
        return None


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
        accessible_cache_path: str | None = None,
        inaccessible_cache_path: str | None = None,
        cache_ttl_seconds: int = 86400,
        timeout: int = 20,
    ):
        self.accessible_cache_path = (
            Path(accessible_cache_path).expanduser() if accessible_cache_path else None
        )
        self.inaccessible_cache_path = (
            Path(inaccessible_cache_path).expanduser()
            if inaccessible_cache_path
            else None
        )
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.timeout = max(1, int(timeout))
        self._accessible_cache = self._load_cache(self.accessible_cache_path)
        self._inaccessible_cache = self._load_cache(self.inaccessible_cache_path)

    def check(self, url: str) -> VideoAvailabilityResult:
        cached = self._get_cached(url, self._accessible_cache, accessible=True)
        if cached is not None:
            return cached

        cached = self._get_cached(url, self._inaccessible_cache, accessible=False)
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
            "logger": _SilentYtDlpLogger(),
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
            return VideoAvailabilityResult(False, self._normalize_youtube_error(exc))

    def _check_direct_url(self, url: str) -> VideoAvailabilityResult:
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code >= 400 or not response.ok:
                return VideoAvailabilityResult(False, f"http_{response.status_code}")
            return VideoAvailabilityResult(True, "http_ok")
        except Exception as exc:
            return VideoAvailabilityResult(False, str(exc))

    def _normalize_youtube_error(self, exc: Exception) -> str:
        message = str(exc).lower()
        if "confirm you’re not a bot" in message or "confirm you're not a bot" in message:
            return "youtube_bot_check"
        if "private video" in message:
            return "youtube_private"
        if "video unavailable" in message:
            return "youtube_unavailable"
        if "sign in" in message:
            return "youtube_sign_in_required"
        if "unsupported url" in message:
            return "youtube_unsupported"
        return "youtube_check_failed"

    def _load_cache(self, cache_path: Path | None) -> dict:
        if cache_path is None or not cache_path.exists():
            return {}

        try:
            return json.loads(cache_path.read_text())
        except Exception:
            return {}

    def _get_cached(
        self, url: str, cache: dict, accessible: bool
    ) -> VideoAvailabilityResult | None:
        entry = cache.get(url)
        if not entry:
            return None

        checked_at = float(entry.get("checked_at", 0))
        if self.cache_ttl_seconds and time.time() - checked_at > self.cache_ttl_seconds:
            cache.pop(url, None)
            return None

        return VideoAvailabilityResult(
            accessible=accessible,
            reason=str(entry.get("reason", "cached")),
        )

    def _store(self, url: str, result: VideoAvailabilityResult) -> None:
        target_cache = (
            self._accessible_cache if result.accessible else self._inaccessible_cache
        )
        other_cache = (
            self._inaccessible_cache if result.accessible else self._accessible_cache
        )
        target_path = (
            self.accessible_cache_path
            if result.accessible
            else self.inaccessible_cache_path
        )
        other_path = (
            self.inaccessible_cache_path
            if result.accessible
            else self.accessible_cache_path
        )

        target_cache[url] = {
            "reason": result.reason,
            "checked_at": time.time(),
        }
        other_cache.pop(url, None)

        self._write_cache(target_path, target_cache)
        self._write_cache(other_path, other_cache)

    def _write_cache(self, cache_path: Path | None, cache: dict) -> None:
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    def refresh_unavailable(self) -> int:
        removed_urls = list(self._inaccessible_cache.keys())
        for url in removed_urls:
            self._inaccessible_cache.pop(url, None)

        self._write_cache(self.inaccessible_cache_path, self._inaccessible_cache)

        return len(removed_urls)

    def get_accessible_urls(self) -> list[str]:
        urls: list[str] = []
        expired_urls: list[str] = []
        now = time.time()
        for url, entry in self._accessible_cache.items():
            checked_at = float(entry.get("checked_at", 0))
            if self.cache_ttl_seconds and now - checked_at > self.cache_ttl_seconds:
                expired_urls.append(url)
                continue
            urls.append(url)

        if expired_urls:
            for url in expired_urls:
                self._accessible_cache.pop(url, None)
            self._write_cache(self.accessible_cache_path, self._accessible_cache)

        return urls
