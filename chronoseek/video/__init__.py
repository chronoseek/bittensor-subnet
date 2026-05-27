"""Shared video acquisition helpers for validators and miners."""

from chronoseek.video.downloader import (
    DownloadAttemptFailure,
    DownloadedVideo,
    VideoDownloadError,
    VideoDownloader,
)

__all__ = [
    "DownloadAttemptFailure",
    "DownloadedVideo",
    "VideoDownloadError",
    "VideoDownloader",
]
