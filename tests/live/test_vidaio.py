"""Live Vidaio compression check."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chronoseek.video.vidaio import VidaioCompressor
from live_helpers import env_value, require_live


pytestmark = pytest.mark.live


def test_live_vidaio_compress() -> None:
    """Compress one real local video file through Vidaio."""

    require_live("CHRONOSEEK_LIVE_VIDAIO", "VIDAIO_API_BASE_URL", "LIVE_VIDAIO_INPUT_PATH")

    input_path = Path(env_value("LIVE_VIDAIO_INPUT_PATH")).expanduser()
    assert input_path.is_file()
    with tempfile.TemporaryDirectory(prefix="chronoseek-live-vidaio-") as tmp_dir:
        output_path = Path(tmp_dir) / "compressed.mp4"
        result = VidaioCompressor(
            api_base_url=env_value("VIDAIO_API_BASE_URL"),
            api_key=env_value("VIDAIO_API_KEY") or None,
            timeout_seconds=float(env_value("VIDAIO_TIMEOUT_SECONDS", "60")),
        ).compress(input_path=str(input_path), output_path=str(output_path))

        assert result.backend == "vidaio"
        assert output_path.exists()
        assert output_path.stat().st_size > 0
