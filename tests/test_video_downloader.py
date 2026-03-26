from chronoseek.miner.utils.video_downloader import VideoDownloader


def test_is_youtube_url_detects_supported_hosts():
    assert VideoDownloader.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert VideoDownloader.is_youtube_url("https://youtu.be/abc123")
    assert not VideoDownloader.is_youtube_url("https://example.com/video.mp4")
