"""Live video downloader check."""

from __future__ import annotations

from pathlib import Path
import os

import pytest
from dotenv import load_dotenv

from chronoseek.miner.cookie_refresh import refresh_cookie_file
from chronoseek.video.downloader import VideoDownloader
from .live_helpers import env_value, require_live


pytestmark = pytest.mark.live
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def test_live_video_download() -> None:
    """Download one real video URL through the shared downloader."""

    # require_live("CHRONOSEEK_LIVE_VIDEO_DOWNLOAD", "LIVE_VIDEO_DOWNLOAD_URL")

    # A standalone downloader test does not enter the miner runtime lifespan,
    # so its periodic refresh worker is not running. Explicitly exercise the
    # endpoint when configured, then verify that yt-dlp consumes the persisted
    # result below.
    if env_value("COOKIE_REFRESH_URL"):
        refreshed_path = refresh_cookie_file(
            target_path=env_value("YTDLP_COOKIES") or None,
        )
        assert Path(refreshed_path).is_file()
        assert os.environ["YTDLP_COOKIES"] == refreshed_path

    downloaded = VideoDownloader.download_video(
        # env_value("LIVE_VIDEO_DOWNLOAD_URL"),
        "https://www.youtube.com/watch?v=aMBA7gG5z-E",
        # timeout=int(env_value("LIVE_VIDEO_DOWNLOAD_TIMEOUT", "120")),
        timeout=120,
        raise_on_failure=True,
    )
    try:
        assert downloaded is not None
        assert Path(downloaded.path).exists()
        assert Path(downloaded.path).stat().st_size > 0
        assert VideoDownloader._looks_like_video_container(downloaded.path)
    finally:
        VideoDownloader.cleanup(downloaded)

if __name__ == "__main__":
    test_live_video_download()
