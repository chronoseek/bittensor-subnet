import os
import tempfile

from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.miner.utils.video_downloader import DownloadedVideo


def test_is_youtube_url_detects_supported_hosts():
    assert VideoDownloader.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert VideoDownloader.is_youtube_url("https://youtu.be/abc123")
    assert not VideoDownloader.is_youtube_url("https://example.com/video.mp4")


def test_cleanup_removes_downloaded_artifacts():
    temp_dir = tempfile.mkdtemp(prefix="chronoseek-test-cleanup-")
    temp_file = os.path.join(temp_dir, "video.mp4")
    with open(temp_file, "wb") as handle:
        handle.write(b"data")

    downloaded_video = DownloadedVideo(
        path=temp_file,
        cleanup_paths=[temp_dir],
    )

    VideoDownloader.cleanup(downloaded_video)

    assert not os.path.exists(temp_dir)
