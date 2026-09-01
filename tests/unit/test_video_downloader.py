import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from chronoseek.hippius.s3 import DEFAULT_HIPPIUS_S3_ENDPOINT_URL
from chronoseek.validator.video_availability import (
    VideoAvailabilityChecker,
    VideoAvailabilityResult,
)
from chronoseek.video.downloader import (
    DownloadedVideo,
    VideoDownloadError,
    VideoDownloader,
)


def test_is_extractor_platform_url_detects_supported_hosts():
    assert VideoDownloader.is_extractor_platform_url(
        "https://www.youtube.com/watch?v=abc123"
    )
    assert VideoDownloader.is_extractor_platform_url("https://youtu.be/abc123")
    assert VideoDownloader.is_extractor_platform_url("https://vimeo.com/12345")
    assert not VideoDownloader.is_extractor_platform_url(
        "https://example.com/video.mp4"
    )


def test_is_hippius_url_detects_default_and_configured_hosts(monkeypatch):
    assert VideoDownloader.is_hippius_url(
        "https://s3.hippius.com/chronoseek/task-clips/task.mp4"
    )
    assert VideoDownloader.is_hippius_url(
        "https://public.hippius.com/chronoseek/task-clips/task.mp4"
    )

    monkeypatch.setattr(
        "chronoseek.video.downloader.DEFAULT_HIPPIUS_S3_PUBLIC_BASE_URL",
        "https://tasks.example.net/chronoseek",
    )
    assert VideoDownloader.is_hippius_url(
        "https://tasks.example.net/chronoseek/task.mp4"
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


def test_download_attempts_route_hippius_before_http():
    attempts = VideoDownloader._download_attempts(
        "https://s3.hippius.com/chronoseek/task-clips/task.mp4"
    )

    assert [name for name, _ in attempts] == ["hippius-s3", "http"]


def test_download_with_hippius_uses_signed_s3_when_credentials_are_configured(
    monkeypatch,
):
    calls = {}

    class FakeHippiusS3StorageClient:
        def __init__(self, config, *, timeout_seconds):
            calls["config"] = config
            calls["timeout_seconds"] = timeout_seconds

        def download_file(self, *, object_key, local_path):
            calls["object_key"] = object_key
            calls["local_path"] = local_path
            Path(local_path).write_bytes(
                b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41"
            )
            return local_path

    monkeypatch.setenv("HIPPIUS_S3_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("HIPPIUS_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(
        "chronoseek.video.downloader.HippiusS3StorageClient",
        FakeHippiusS3StorageClient,
    )

    downloaded = VideoDownloader._download_with_hippius(
        "https://s3.hippius.com/chronoseek/task-clips/demo-task.mp4",
        timeout=7,
    )

    try:
        assert Path(downloaded.path).read_bytes().startswith(b"\x00\x00\x00\x14ftyp")
        assert downloaded.cleanup_paths == [downloaded.path]
        assert calls["object_key"] == "task-clips/demo-task.mp4"
        assert calls["local_path"] == downloaded.path
        assert calls["timeout_seconds"] == 7.0
        assert calls["config"].endpoint_url == DEFAULT_HIPPIUS_S3_ENDPOINT_URL
        assert calls["config"].public_base_url == DEFAULT_HIPPIUS_S3_ENDPOINT_URL
        assert calls["config"].bucket == "chronoseek"
        assert calls["config"].access_key_id == "access"
        assert calls["config"].secret_access_key == "secret"
        assert calls["config"].region == "decentralized"
        assert calls["config"].anonymous is False
    finally:
        VideoDownloader.cleanup(downloaded)


def test_download_with_hippius_uses_anonymous_minio_without_credentials(monkeypatch):
    calls = {}

    class FakeHippiusS3StorageClient:
        def __init__(self, config, *, timeout_seconds):
            calls["config"] = config
            calls["timeout_seconds"] = timeout_seconds

        def download_file(self, *, object_key, local_path):
            calls["object_key"] = object_key
            calls["local_path"] = local_path
            Path(local_path).write_bytes(
                b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41"
            )
            return local_path

    monkeypatch.delenv("HIPPIUS_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("HIPPIUS_S3_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(
        "chronoseek.video.downloader.HippiusS3StorageClient",
        FakeHippiusS3StorageClient,
    )

    downloaded = VideoDownloader._download_with_hippius(
        "https://s3.hippius.com/chronoseek/task-clips/task.mp4",
        timeout=9,
    )

    try:
        assert Path(downloaded.path).read_bytes().startswith(b"\x00\x00\x00\x14ftyp")
        assert calls["object_key"] == "task-clips/task.mp4"
        assert calls["local_path"] == downloaded.path
        assert calls["timeout_seconds"] == 9.0
        assert calls["config"].bucket == "chronoseek"
        assert calls["config"].access_key_id == ""
        assert calls["config"].secret_access_key == ""
        assert calls["config"].anonymous is True
    finally:
        VideoDownloader.cleanup(downloaded)


def test_validator_availability_routes_vimeo_through_extractor(monkeypatch):
    checker = VideoAvailabilityChecker(cache_ttl_seconds=0)
    calls = []

    def fake_extractor_check(url):
        calls.append(("extractor", url))
        return VideoAvailabilityResult(True, "extractor_ok")

    def fake_direct_check(url):
        calls.append(("direct", url))
        return VideoAvailabilityResult(True, "http_ok")

    monkeypatch.setattr(checker, "_check_extractor_platform", fake_extractor_check)
    monkeypatch.setattr(checker, "_check_direct_url", fake_direct_check)

    result = checker.check("https://vimeo.com/12345")

    assert result.accessible
    assert result.reason == "extractor_ok"
    assert calls == [("extractor", "https://vimeo.com/12345")]


@patch("yt_dlp.YoutubeDL")
def test_check_extractor_platform_reuses_downloader_cookie_and_player_client_config(
    mock_youtube_dl_cls, monkeypatch, tmp_path
):
    """Regression test: the availability pre-check built its own bare yt-dlp
    options with no cookies at all, so it hit YouTube's bot-check on almost
    every candidate even when the real download (which does carry cookies)
    would have succeeded - wasting sampling attempts on videos that were
    actually accessible."""
    cookies_file = tmp_path / "cookies.txt"
    cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES", str(cookies_file))
    monkeypatch.delenv("YTDLP_COOKIES_BROWSER", raising=False)

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "abc123"}
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    captured_cookie_contents = {}

    def capture_options(options):
        captured_cookie_contents["path"] = options["cookiefile"]
        captured_cookie_contents["contents"] = Path(
            options["cookiefile"]
        ).read_text(encoding="utf-8")
        captured_cookie_contents["player_client"] = options["extractor_args"][
            "youtube"
        ]["player_client"]
        return mock_youtube_dl_cls.return_value

    mock_youtube_dl_cls.side_effect = capture_options

    checker = VideoAvailabilityChecker(cache_ttl_seconds=0)
    result = checker._check_extractor_platform(
        "https://www.youtube.com/watch?v=abc123"
    )

    assert result.accessible
    assert captured_cookie_contents["player_client"] == (
        VideoDownloader.YTDLP_PLAYER_CLIENTS
    )
    assert captured_cookie_contents["path"] != str(cookies_file)
    assert captured_cookie_contents["contents"] == "# Netscape HTTP Cookie File\n"


def test_ytdlp_cookie_options_uses_cookie_file_when_present(monkeypatch, tmp_path):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.delenv("YTDLP_COOKIES_BROWSER", raising=False)
    monkeypatch.setenv("YTDLP_COOKIES", str(cookies))
    opts = VideoDownloader._ytdlp_cookie_options()
    assert opts == {"cookiefile": str(cookies)}


def test_ytdlp_cookie_options_prefers_file_over_browser(monkeypatch, tmp_path):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES", str(cookies))
    monkeypatch.setenv("YTDLP_COOKIES_BROWSER", "chrome:Default")
    opts = VideoDownloader._ytdlp_cookie_options()
    assert "cookiefile" in opts
    assert "cookiesfrombrowser" not in opts


def test_download_with_ytdlp_uses_private_writable_cookie_copy(
    monkeypatch,
    tmp_path,
):
    import yt_dlp

    source_cookies = tmp_path / "cookies.txt"
    source_cookies.write_text(
        "# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tFALSE\t0\tname\tvalue\n",
        encoding="utf-8",
    )
    source_cookies.chmod(0o444)
    monkeypatch.setenv("YTDLP_COOKIES", str(source_cookies))
    monkeypatch.delenv("YTDLP_COOKIES_BROWSER", raising=False)
    captured = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            cookie_path = Path(options["cookiefile"])
            captured["cookie_path"] = cookie_path
            captured["cookie_contents"] = cookie_path.read_text(encoding="utf-8")
            captured["cookie_mode"] = cookie_path.stat().st_mode & 0o777

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            Path(self.options["cookiefile"]).write_text(
                "# updated by yt-dlp\n",
                encoding="utf-8",
            )

        def extract_info(self, url, download):
            return {"id": "video", "ext": "mp4"}

        def prepare_filename(self, info):
            output_path = Path(self.options["outtmpl"] % info)
            output_path.write_bytes(
                b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41"
            )
            return str(output_path)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    downloaded = VideoDownloader._download_with_ytdlp(
        "https://www.youtube.com/watch?v=example",
        timeout=10,
    )

    try:
        runtime_cookie_path = captured["cookie_path"]
        assert runtime_cookie_path != source_cookies
        assert runtime_cookie_path.parent == Path(downloaded.cleanup_paths[0])
        assert captured["cookie_contents"] == source_cookies.read_text(
            encoding="utf-8"
        )
        assert captured["cookie_mode"] == 0o600
        assert source_cookies.stat().st_mode & 0o777 == 0o444
    finally:
        VideoDownloader.cleanup(downloaded)


def test_download_with_ytdlp_refreshes_cookies_and_retries_once(
    monkeypatch,
    tmp_path,
):
    import yt_dlp
    import chronoseek.video.downloader as downloader_module

    refreshed_cookies = tmp_path / "fresh-cookies.txt"
    refreshed_cookies.write_text("# Netscape HTTP Cookie File\nfresh\n", encoding="utf-8")
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_BROWSER", raising=False)
    monkeypatch.setenv("COOKIE_REFRESH_URL", "https://miner.example")
    refresh_calls = []

    def fake_refresh_cookie_file(**kwargs):
        refresh_calls.append(kwargs)
        return str(refreshed_cookies)

    monkeypatch.setattr(
        downloader_module,
        "refresh_cookie_file",
        fake_refresh_cookie_file,
    )
    attempts = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options
            attempts.append(dict(options))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def extract_info(self, url, download):
            if len(attempts) == 1:
                raise RuntimeError("Sign in to confirm you're not a bot")
            return {"id": "video", "ext": "mp4"}

        def prepare_filename(self, info):
            output_path = Path(self.options["outtmpl"] % info)
            output_path.write_bytes(
                b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isomiso2mp41"
            )
            return str(output_path)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    downloaded = VideoDownloader._download_with_ytdlp(
        "https://www.youtube.com/watch?v=example",
        timeout=10,
    )
    try:
        assert len(attempts) == 2
        assert refresh_calls == [{"target_path": None}]
        assert "cookiefile" not in attempts[0]
        assert attempts[1]["cookiefile"] == str(refreshed_cookies)
        assert Path(downloaded.path).is_file()
    finally:
        VideoDownloader.cleanup(downloaded)


def test_ytdlp_cookie_options_is_empty_without_cookie_config(monkeypatch):
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YTDLP_COOKIES_BROWSER", raising=False)

    assert VideoDownloader._ytdlp_cookie_options() == {}


def test_ytdlp_js_runtime_options_uses_configured_paths(monkeypatch, tmp_path):
    node = tmp_path / "node"
    deno = tmp_path / "deno"
    monkeypatch.setenv("YTDLP_NODE_PATH", str(node))
    monkeypatch.setenv("YTDLP_DENO_PATH", str(deno))

    assert VideoDownloader._ytdlp_js_runtime_options() == {
        "js_runtimes": {
            "node": {"path": str(node)},
            "deno": {"path": str(deno)},
        },
        "remote_components": ["ejs:npm"],
    }


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
@patch.object(VideoDownloader, "_download_with_hippius")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_prefers_hippius_for_hippius_urls(
    mock_download_with_requests,
    mock_download_with_hippius,
    _mock_looks_like_video_file,
    _mock_looks_like_video_container,
):
    downloaded_video = DownloadedVideo(path="/tmp/test.mp4", cleanup_paths=["/tmp/test.mp4"])
    mock_download_with_hippius.return_value = downloaded_video

    result = VideoDownloader.download_video(
        "https://s3.hippius.com/chronoseek/task-clips/task.mp4"
    )

    assert result == downloaded_video
    mock_download_with_hippius.assert_called_once()
    mock_download_with_requests.assert_not_called()


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


@patch.object(VideoDownloader, "_download_with_ytdlp")
@patch.object(VideoDownloader, "_download_with_requests")
def test_download_video_can_raise_structured_failure(
    mock_download_with_requests,
    mock_download_with_ytdlp,
):
    mock_download_with_ytdlp.side_effect = RuntimeError("extractor failed")
    mock_download_with_requests.side_effect = RuntimeError("http failed")

    try:
        VideoDownloader.download_video(
            "https://example.com/watch?v=abc123",
            raise_on_failure=True,
        )
    except VideoDownloadError as exc:
        assert exc.details() == {
            "video_url": "https://example.com/watch?v=abc123",
            "download_attempts": [
                {"backend": "yt-dlp", "message": "extractor failed"},
                {"backend": "http", "message": "http failed"},
            ],
        }
    else:
        raise AssertionError("expected structured download failure")


@patch.object(VideoDownloader, "_download_with_requests")
@patch.object(VideoDownloader, "_download_with_ytdlp")
def test_download_video_youtube_watch_url_falls_back_to_http_after_ytdlp_failure(
    mock_download_with_ytdlp,
    mock_download_with_requests,
):
    mock_download_with_ytdlp.side_effect = RuntimeError("Sign in to confirm")
    mock_download_with_requests.side_effect = RuntimeError(
        "URL returned non-media Content-Type: text/html"
    )

    result = VideoDownloader.download_video("https://www.youtube.com/watch?v=abc123")

    assert result is None
    mock_download_with_ytdlp.assert_called_once()
    mock_download_with_requests.assert_called_once()


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


@patch("yt_dlp.YoutubeDL")
def test_download_with_ytdlp_excludes_av01_format(mock_youtube_dl_cls, tmp_path):
    """Regression test: OpenCV's bundled FFmpeg fails to decode AV1 on some
    platforms/builds ("Failed to get pixel format") even though the system
    ffmpeg CLI decodes the same file fine via libdav1d, so the format
    selector must not let yt-dlp pick an AV1-coded stream unless nothing
    else is available."""
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
    mock_ydl.prepare_filename.return_value = str(tmp_path / "abc123.mp4")
    mock_youtube_dl_cls.return_value.__enter__.return_value = mock_ydl

    VideoDownloader._download_with_ytdlp(
        "https://www.youtube.com/watch?v=abc123", timeout=30
    )

    options = mock_youtube_dl_cls.call_args.args[0]
    assert options["format"] == (
        "mp4[vcodec!*=av01]/bestvideo[vcodec!*=av01]+bestaudio"
        "/best[vcodec!*=av01]/bestvideo+bestaudio/best"
    )


def test_ytdlp_clean_parent_env_strips_node_ipc_vars(monkeypatch):
    monkeypatch.setenv("NODE_CHANNEL_FD", "99")
    monkeypatch.setenv("NODE_CHANNEL_SERIALIZATION_MODE", "json")
    with VideoDownloader._yt_dlp_clean_parent_env():
        assert "NODE_CHANNEL_FD" not in os.environ
        assert "NODE_CHANNEL_SERIALIZATION_MODE" not in os.environ
    assert os.environ.get("NODE_CHANNEL_FD") == "99"
    assert os.environ.get("NODE_CHANNEL_SERIALIZATION_MODE") == "json"
