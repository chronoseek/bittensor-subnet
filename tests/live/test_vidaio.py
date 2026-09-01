"""Live Vidaio compression check."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chronoseek.constants import DEFAULT_VIDAIO_API_BASE_URL
from chronoseek.video.vidaio import VidaioCompressor
from live_helpers import env_value, require_live


pytestmark = pytest.mark.live


def test_live_vidaio_compress() -> None:
    """Compress one public video URL through Vidaio."""

    require_live("CHRONOSEEK_LIVE_VIDAIO", "VIDAIO_API_KEY", "LIVE_VIDAIO_INPUT_URL")

    output_path = Path(tempfile.gettempdir()) / "chronoseek-live-vidaio-unused.mp4"
    result = VidaioCompressor(
        api_base_url=DEFAULT_VIDAIO_API_BASE_URL,
        api_key=env_value("VIDAIO_API_KEY") or None,
        timeout_seconds=float(env_value("VIDAIO_TIMEOUT_SECONDS", "60")),
        poll_interval_seconds=float(env_value("VIDAIO_POLL_INTERVAL_SECONDS", "15")),
    ).compress_url(
        video_url=env_value("LIVE_VIDAIO_INPUT_URL"),
        output_path=str(output_path),
    )

    assert result.backend == "vidaio"
    assert result.public_url.startswith(("http://", "https://"))
    assert result.path == ""
