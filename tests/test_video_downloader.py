import os
import tempfile
from unittest.mock import patch

from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.miner.utils.video_downloader import DownloadedVideo


def test_is_extractor_platform_url_detects_supported_hosts():
    assert VideoDownloader.is_extractor_platform_url(
        "https://www.youtube.com/watch?v=abc123"
    )
    assert VideoDownloader.is_extractor_platform_url("https://youtu.be/abc123")
    assert VideoDownloader.is_extractor_platform_url("https://vimeo.com/12345")
    assert not VideoDownloader.is_extractor_platform_url(
        "https://example.com/video.mp4"
    )


def test_is_direct_media_url_detects_file_urls():
    assert VideoDownloader.is_direct_media_url("https://example.com/video.mp4")
    assert VideoDownloader.is_direct_media_url("https://cdn.example.com/path/clip.webm")
    assert not VideoDownloader.is_direct_media_url("https://example.com/watch?v=abc123")


def test_should_prefer_ytdlp_for_platform_and_ambiguous_urls():
    assert VideoDownloader.should_prefer_ytdlp("https://www.youtube.com/watch?v=abc123")
    assert VideoDownloader.should_prefer_ytdlp("https://vimeo.com/12345")
    assert VideoDownloader.should_prefer_ytdlp("https://example.com/watch?v=abc123")
    assert not VideoDownloader.should_prefer_ytdlp(
        "https://example.com/video.mp4"
    )


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


@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_prefers_http_for_direct_media_urls(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_requests.return_value = downloaded_video

    result = VideoDownloader.download_video("https://example.com/video.mp4")

    assert result == downloaded_video
    mock_download_with_requests.assert_called_once()
    mock_download_with_ytdlp.assert_not_called()


@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_prefers_ytdlp_for_ambiguous_urls(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_ytdlp.return_value = downloaded_video

    result = VideoDownloader.download_video("https://example.com/watch?v=abc123")

    assert result == downloaded_video
    mock_download_with_ytdlp.assert_called_once()
    mock_download_with_requests.assert_not_called()


@patch.object(VideoDownloader, "cleanup")
@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_falls_back_when_primary_strategy_fails(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
    mock_cleanup,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_requests.side_effect = RuntimeError("http failed")
    mock_download_with_ytdlp.return_value = downloaded_video

    result = VideoDownloader.download_video("https://example.com/video.mp4")

    assert result == downloaded_video
    mock_download_with_requests.assert_called_once()
    mock_download_with_ytdlp.assert_called_once()
    mock_cleanup.assert_called()
