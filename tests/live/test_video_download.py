"""Live video downloader check."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronoseek.video.downloader import VideoDownloader
from live_helpers import env_value, require_live


pytestmark = pytest.mark.live


def test_live_video_download() -> None:
    """Download one real video URL through the shared downloader."""

    require_live("CHRONOSEEK_LIVE_VIDEO_DOWNLOAD", "LIVE_VIDEO_DOWNLOAD_URL")

    downloaded = VideoDownloader.download_video(
        env_value("LIVE_VIDEO_DOWNLOAD_URL"),
        timeout=int(env_value("LIVE_VIDEO_DOWNLOAD_TIMEOUT", "120")),
        raise_on_failure=True,
    )
    try:
        assert downloaded is not None
        assert Path(downloaded.path).exists()
        assert Path(downloaded.path).stat().st_size > 0
        assert VideoDownloader._looks_like_video_container(downloaded.path)
    finally:
        VideoDownloader.cleanup(downloaded)
