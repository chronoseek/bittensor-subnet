import shutil
import subprocess
from unittest.mock import patch

import pytest

from chronoseek.miner.utils.frame_extractor import FrameExtractor


def _encode_test_clip(path: str, *, vcodec: str, extra_args: list[str] | None = None) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x64:rate=5",
            "-c:v",
            vcodec,
            *(extra_args or []),
            path,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "h264.mp4")
    _encode_test_clip(path, vcodec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    return path


@pytest.fixture(scope="module")
def av1_clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "av1.mp4")
    _encode_test_clip(path, vcodec="libaom-av1", extra_args=["-strict", "experimental"])
    return path


def test_extract_frames_decodes_h264_without_transcoding(h264_clip):
    """A directly-decodable video shouldn't pay for a transcode."""
    with patch.object(
        FrameExtractor, "_transcode_to_h264", wraps=FrameExtractor._transcode_to_h264
    ) as mock_transcode:
        frames = FrameExtractor.extract_frames(h264_clip, fps=1)

    assert len(frames) > 0
    mock_transcode.assert_not_called()


def test_extract_frames_transcodes_av1_and_still_returns_frames(av1_clip):
    """Regression test: OpenCV's bundled FFmpeg fails to decode this exact
    AV1 test clip ("Failed to get pixel format") even though the system
    ffmpeg CLI decodes it fine via libdav1d - this reproduced a real miner
    failure (VIDEO_UNREADABLE) on a Vidaio-compressed validator task clip
    that was genuinely AV1-encoded despite being labeled h264."""
    with patch.object(
        FrameExtractor, "_transcode_to_h264", wraps=FrameExtractor._transcode_to_h264
    ) as mock_transcode:
        frames = FrameExtractor.extract_frames(av1_clip, fps=1)

    assert len(frames) > 0
    mock_transcode.assert_called_once_with(av1_clip)


def test_extract_frames_in_windows_transcodes_av1_and_still_returns_frames(av1_clip):
    frames = FrameExtractor.extract_frames_in_windows(av1_clip, [(0.0, 1.0)], fps=2)

    assert len(frames) > 0


def test_extract_frames_returns_empty_when_ffmpeg_binary_is_unavailable(av1_clip):
    with patch("chronoseek.miner.utils.frame_extractor.shutil.which", return_value=None):
        frames = FrameExtractor.extract_frames(av1_clip, fps=1)

    assert frames == []


def test_extract_frames_returns_empty_for_a_genuinely_corrupt_file(tmp_path):
    bad_path = tmp_path / "not_a_video.mp4"
    bad_path.write_bytes(b"this is not a real video file")

    frames = FrameExtractor.extract_frames(str(bad_path), fps=1)

    assert frames == []


def test_transcode_to_h264_cleans_up_and_returns_none_on_failure(tmp_path):
    bad_path = tmp_path / "not_a_video.mp4"
    bad_path.write_bytes(b"this is not a real video file")

    result = FrameExtractor._transcode_to_h264(str(bad_path))

    assert result is None
