"""Live Hippius S3 upload/download check."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from chronoseek.hippius.s3 import HippiusS3Config, HippiusS3StorageClient
from chronoseek.video.downloader import VideoDownloader
from live_helpers import env_bool, env_value, require_live


pytestmark = pytest.mark.live

DEFAULT_OBJECT_KEY = f"task-clips/live-test-9265ab09a75441de918cd45a6c4589b2.mp4"
UPLOADED_PUBLIC_URL: str | None = None
UPLOADED_OBJECT_KEY: str | None = None
UPLOADED_SHA256: str | None = None
UPLOADED_SIZE: int | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def live_hippius_source_video_url() -> str:
    source_url = env_value("LIVE_HIPPIUS_SOURCE_VIDEO_URL") or env_value(
        "LIVE_VIDEO_DOWNLOAD_URL"
    )
    if not source_url:
        pytest.skip(
            "set LIVE_HIPPIUS_SOURCE_VIDEO_URL or LIVE_VIDEO_DOWNLOAD_URL "
            "to run the Hippius upload/download live test"
        )
    return source_url


def live_hippius_bucket() -> str:
    return env_value("HIPPIUS_S3_BUCKET", "chronoseek")


def live_hippius_object_key() -> str:
    return env_value("LIVE_HIPPIUS_OBJECT_KEY", DEFAULT_OBJECT_KEY)


def live_hippius_client(*, anonymous: bool = False) -> HippiusS3StorageClient:
    return HippiusS3StorageClient(
        HippiusS3Config(
            endpoint_url=env_value("HIPPIUS_S3_ENDPOINT_URL", "https://s3.hippius.com"),
            public_base_url=env_value(
                "HIPPIUS_S3_PUBLIC_BASE_URL",
                "https://s3.hippius.com",
            ),
            bucket=live_hippius_bucket(),
            access_key_id="" if anonymous else env_value("HIPPIUS_S3_ACCESS_KEY_ID"),
            secret_access_key=(
                "" if anonymous else env_value("HIPPIUS_S3_SECRET_ACCESS_KEY")
            ),
            region=env_value("HIPPIUS_S3_REGION", "decentralized"),
            anonymous=anonymous,
        ),
        timeout_seconds=float(env_value("HIPPIUS_S3_TIMEOUT_SECONDS", "60")),
    )


def live_hippius_public_url() -> str:
    if env_value("LIVE_HIPPIUS_PUBLIC_URL"):
        return env_value("LIVE_HIPPIUS_PUBLIC_URL")
    if UPLOADED_PUBLIC_URL:
        return UPLOADED_PUBLIC_URL
    if env_value("LIVE_HIPPIUS_OBJECT_KEY"):
        return live_hippius_client(anonymous=True).public_url(live_hippius_object_key())
    pytest.skip(
        "run test_01_live_hippius_upload first, or set LIVE_HIPPIUS_PUBLIC_URL "
        "or LIVE_HIPPIUS_OBJECT_KEY for a pre-existing public Hippius object"
    )


def test_01_live_hippius_upload() -> None:
    """Step 1: ensure public bucket, then upload with validator credentials."""

    require_live(
        "CHRONOSEEK_LIVE_HIPPIUS",
        "HIPPIUS_S3_ACCESS_KEY_ID",
        "HIPPIUS_S3_SECRET_ACCESS_KEY",
    )

    global UPLOADED_OBJECT_KEY, UPLOADED_PUBLIC_URL, UPLOADED_SHA256, UPLOADED_SIZE

    source_url = live_hippius_source_video_url()
    bucket = live_hippius_bucket()
    object_key = live_hippius_object_key()
    upload_client = live_hippius_client()
    upload_client.ensure_bucket_public()

    downloaded = None
    try:
        downloaded = VideoDownloader.download_video(
            source_url,
            timeout=int(env_value("LIVE_HIPPIUS_SOURCE_DOWNLOAD_TIMEOUT", "120")),
            raise_on_failure=True,
        )
        assert downloaded is not None
        upload_path = Path(downloaded.path)
        assert upload_path.exists()
        assert upload_path.stat().st_size > 0
        assert VideoDownloader._looks_like_video_container(str(upload_path))

        public_url = upload_client.upload_file(
            local_path=str(upload_path),
            object_key=object_key,
            content_type="video/mp4",
        )

        UPLOADED_OBJECT_KEY = object_key
        UPLOADED_PUBLIC_URL = public_url
        UPLOADED_SHA256 = file_sha256(upload_path)
        UPLOADED_SIZE = upload_path.stat().st_size

        assert public_url.endswith(f"/{bucket}/{object_key}")
        print(f"LIVE_HIPPIUS_OBJECT_KEY={object_key}")
        print(f"LIVE_HIPPIUS_PUBLIC_URL={public_url}")
    finally:
        VideoDownloader.cleanup(downloaded)


def test_02_live_hippius_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 2: download the public Hippius URL without Hippius credentials."""

    require_live("CHRONOSEEK_LIVE_HIPPIUS")

    public_url = live_hippius_public_url()
    hippius_downloaded = None
    try:
        with monkeypatch.context() as env_without_hippius_credentials:
            env_without_hippius_credentials.delenv(
                "HIPPIUS_S3_ACCESS_KEY_ID",
                raising=False,
            )
            env_without_hippius_credentials.delenv(
                "HIPPIUS_S3_SECRET_ACCESS_KEY",
                raising=False,
            )
            assert not env_value("HIPPIUS_S3_ACCESS_KEY_ID")
            assert not env_value("HIPPIUS_S3_SECRET_ACCESS_KEY")

            # Miners must be able to fetch validator-uploaded Hippius task clips
            # without inheriting validator-side Hippius credentials.
            hippius_downloaded = VideoDownloader.download_video(
                public_url,
                timeout=int(env_value("LIVE_HIPPIUS_DOWNLOAD_TIMEOUT", "120")),
                raise_on_failure=True,
            )
        assert hippius_downloaded is not None
        download_path = Path(hippius_downloaded.path)

        assert download_path.exists()
        assert download_path.stat().st_size > 0
        if UPLOADED_SIZE is not None:
            assert download_path.stat().st_size == UPLOADED_SIZE
        if UPLOADED_SHA256 is not None:
            assert file_sha256(download_path) == UPLOADED_SHA256
        assert VideoDownloader._looks_like_video_container(str(download_path))
    finally:
        VideoDownloader.cleanup(hippius_downloaded)
