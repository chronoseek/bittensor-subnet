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


def test_ytdlp_cookie_options_uses_cookie_file_when_present(monkeypatch, tmp_path):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("CHRONOSEEK_YTDLP_COOKIES", str(cookies))
    monkeypatch.delenv("CHRONOSEEK_YTDLP_COOKIES_BROWSER", raising=False)
    opts = VideoDownloader._ytdlp_cookie_options()
    assert opts == {"cookiefile": str(cookies)}


def test_ytdlp_cookie_options_prefers_file_over_browser(monkeypatch, tmp_path):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("CHRONOSEEK_YTDLP_COOKIES", str(cookies))
    monkeypatch.setenv("CHRONOSEEK_YTDLP_COOKIES_BROWSER", "chrome")
    opts = VideoDownloader._ytdlp_cookie_options()
    assert "cookiefile" in opts
    assert "cookiesfrombrowser" not in opts


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


@patch.object(VideoDownloader, "_looks_like_video_container", return_value=True)
@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_prefers_http_for_direct_media_urls(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
    _mock_looks_like_video_container,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_requests.return_value = downloaded_video

    result = VideoDownloader.download_video("https://example.com/video.mp4")

    assert result == downloaded_video
    mock_download_with_requests.assert_called_once()
    mock_download_with_ytdlp.assert_not_called()


@patch.object(VideoDownloader, "_looks_like_video_container", return_value=True)
@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_prefers_ytdlp_for_ambiguous_urls(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
    _mock_looks_like_video_container,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_ytdlp.return_value = downloaded_video

    result = VideoDownloader.download_video("https://example.com/watch?v=abc123")

    assert result == downloaded_video
    mock_download_with_ytdlp.assert_called_once()
    mock_download_with_requests.assert_not_called()


@patch.object(VideoDownloader, "cleanup")
@patch.object(VideoDownloader, "_looks_like_video_container", return_value=True)
@patch.object(VideoDownloader, "_looks_like_video_file", return_value=True)
@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_falls_back_when_primary_strategy_fails(
    mock_download_with_requests,
    mock_download_with_ytdlp,
    _mock_looks_like_video_file,
    _mock_looks_like_video_container,
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


@patch.object(VideoDownloader, "_download_with_requests")
@patch.object(VideoDownloader, "_download_with_ytdlp")
def test_download_video_youtube_watch_url_has_no_http_fallback(
    mock_download_with_ytdlp,
    mock_download_with_requests,
):
    mock_download_with_ytdlp.side_effect = RuntimeError("Sign in to confirm")

    result = VideoDownloader.download_video("https://www.youtube.com/watch?v=abc123")

    assert result is None
    mock_download_with_ytdlp.assert_called_once()
    mock_download_with_requests.assert_not_called()


def test_looks_like_video_container_accepts_mp4_and_rejects_html(tmp_path):
    good = tmp_path / "real.mp4"
    # Minimal ftyp box: size 20, 'ftyp', brand 'isom'
    good.write_bytes(
        b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41"
    )
    assert VideoDownloader._looks_like_video_container(str(good))

    bad = tmp_path / "fake.mp4"
    bad.write_text("<!DOCTYPE html><html>", encoding="utf-8")
    assert not VideoDownloader._looks_like_video_container(str(bad))


def test_ytdlp_clean_parent_env_strips_node_ipc_vars(monkeypatch):
    monkeypatch.setenv("NODE_CHANNEL_FD", "99")
    monkeypatch.setenv("NODE_CHANNEL_SERIALIZATION_MODE", "json")
    with VideoDownloader._yt_dlp_clean_parent_env():
        assert "NODE_CHANNEL_FD" not in os.environ
        assert "NODE_CHANNEL_SERIALIZATION_MODE" not in os.environ
    assert os.environ.get("NODE_CHANNEL_FD") == "99"
    assert os.environ.get("NODE_CHANNEL_SERIALIZATION_MODE") == "json"

