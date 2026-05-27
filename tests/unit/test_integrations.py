import json
from pathlib import Path

import pytest
from minio.error import S3Error

from chronoseek.hippius.s3 import (
    HippiusS3Config,
    HippiusS3ObjectRef,
    HippiusS3StorageClient,
    parse_hippius_s3_url,
)
from chronoseek.validator.compression import CompositeCompressor, CompressionResult
from chronoseek.video.vidaio import VidaioCompressor


def test_hippius_storage_public_url_and_config_validation():
    client = HippiusS3StorageClient(
        HippiusS3Config(
            endpoint_url="https://s3.example",
            public_base_url="https://public.example/base",
            bucket="validator-tasks",
            access_key_id="access",
            secret_access_key="secret",
        )
    )

    assert (
        client.public_url("task clips/demo task.mp4")
        == "https://public.example/base/task%20clips/demo%20task.mp4"
    )

    with pytest.raises(ValueError, match="bucket"):
        HippiusS3StorageClient(
            HippiusS3Config(
                endpoint_url="https://s3.example",
                public_base_url="https://public.example/base",
                bucket="",
                access_key_id="access",
                secret_access_key="secret",
            )
        )


def test_hippius_storage_uses_documented_path_style_public_urls():
    client = HippiusS3StorageClient(
        HippiusS3Config(
            access_key_id="access",
            secret_access_key="secret",
        )
    )

    assert client.config.region == "decentralized"
    assert client.config.bucket == "chronoseek"
    assert client.public_url("task-clips/demo-task.mp4") == (
        "https://s3.hippius.com/chronoseek/task-clips/demo-task.mp4"
    )


def test_parse_hippius_s3_url_decodes_documented_path_style_url():
    ref = parse_hippius_s3_url(
        "https://s3.hippius.com/chronoseek/task-clips/demo-task.mp4"
    )

    assert ref == HippiusS3ObjectRef(
        bucket="chronoseek",
        object_key="task-clips/demo-task.mp4",
    )


def test_parse_hippius_s3_url_supports_custom_public_base():
    ref = parse_hippius_s3_url(
        "https://tasks.example.net/chronoseek/task-clips/demo-task.mp4",
        default_bucket="chronoseek",
        public_base_url="https://tasks.example.net/chronoseek",
    )

    assert ref == HippiusS3ObjectRef(
        bucket="chronoseek",
        object_key="task-clips/demo-task.mp4",
    )


def test_hippius_storage_uses_minio_for_upload_download_delete(monkeypatch, tmp_path):
    captured = {}

    class FakeMinio:
        def __init__(
            self,
            endpoint,
            *,
            access_key,
            secret_key,
            secure,
            region,
            http_client,
        ):
            captured["client"] = {
                "endpoint": endpoint,
                "access_key": access_key,
                "secret_key": secret_key,
                "secure": secure,
                "region": region,
                "has_http_client": http_client is not None,
            }

        def fput_object(self, bucket_name, object_name, file_path, *, content_type):
            captured["upload"] = {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_path": file_path,
                "content_type": content_type,
            }

        def fget_object(self, bucket_name, object_name, file_path):
            captured["download"] = {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_path": file_path,
            }
            Path(file_path).write_bytes(b"video-bytes")

        def remove_object(self, bucket_name, object_name):
            captured["delete"] = {
                "bucket_name": bucket_name,
                "object_name": object_name,
            }

    monkeypatch.setattr("chronoseek.hippius.s3.Minio", FakeMinio)

    client = HippiusS3StorageClient(
        HippiusS3Config(
            access_key_id="access",
            secret_access_key="secret",
        ),
        timeout_seconds=12,
    )
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "downloaded.mp4"
    input_path.write_bytes(b"upload-bytes")

    public_url = client.upload_file(
        local_path=str(input_path),
        object_key="task-clips/demo-task.mp4",
        content_type="video/mp4",
    )
    result = client.download_file(
        object_key="task-clips/demo-task.mp4",
        local_path=str(output_path),
    )
    client.delete_object("task-clips/demo-task.mp4")

    assert captured["client"] == {
        "endpoint": "s3.hippius.com",
        "access_key": "access",
        "secret_key": "secret",
        "secure": True,
        "region": "decentralized",
        "has_http_client": True,
    }
    assert public_url == (
        "https://s3.hippius.com/chronoseek/task-clips/demo-task.mp4"
    )
    assert result == str(output_path)
    assert output_path.read_bytes() == b"video-bytes"
    assert captured["upload"] == {
        "bucket_name": "chronoseek",
        "object_name": "task-clips/demo-task.mp4",
        "file_path": str(input_path),
        "content_type": "video/mp4",
    }
    assert captured["download"] == {
        "bucket_name": "chronoseek",
        "object_name": "task-clips/demo-task.mp4",
        "file_path": str(output_path),
    }
    assert captured["delete"] == {
        "bucket_name": "chronoseek",
        "object_name": "task-clips/demo-task.mp4",
    }


def test_hippius_storage_ensures_bucket_exists_and_public(monkeypatch):
    captured = {}

    class FakeMinio:
        def __init__(self, *args, **kwargs):
            return None

        def bucket_exists(self, bucket_name):
            captured["bucket_exists"] = bucket_name
            return False

        def make_bucket(self, bucket_name):
            captured["make_bucket"] = bucket_name

        def set_bucket_policy(self, bucket_name, policy):
            captured["set_bucket_policy"] = {
                "bucket_name": bucket_name,
                "policy": json.loads(policy),
            }

    monkeypatch.setattr("chronoseek.hippius.s3.Minio", FakeMinio)

    client = HippiusS3StorageClient(
        HippiusS3Config(
            bucket="chronoseek",
            access_key_id="access",
            secret_access_key="secret",
        )
    )
    client.ensure_bucket_public()

    assert captured["bucket_exists"] == "chronoseek"
    assert captured["make_bucket"] == "chronoseek"
    assert captured["set_bucket_policy"] == {
        "bucket_name": "chronoseek",
        "policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": ["arn:aws:s3:::chronoseek/*"],
                }
            ],
        },
    }


def test_hippius_storage_accepts_existing_public_bucket_policy(monkeypatch):
    captured = {}

    class FakeMinio:
        def __init__(self, *args, **kwargs):
            return None

        def bucket_exists(self, bucket_name):
            captured["bucket_exists"] = bucket_name
            return True

        def set_bucket_policy(self, bucket_name, policy):
            captured["set_bucket_policy"] = bucket_name
            raise S3Error(
                None,
                "PolicyAlreadyExists",
                "The bucket policy already exists and bucket is public",
                None,
                "request-id",
                "hippius-s3",
                bucket_name=bucket_name,
            )

    monkeypatch.setattr("chronoseek.hippius.s3.Minio", FakeMinio)

    client = HippiusS3StorageClient(
        HippiusS3Config(
            bucket="chronoseek",
            access_key_id="access",
            secret_access_key="secret",
        )
    )
    client.ensure_bucket_public()

    assert captured == {
        "bucket_exists": "chronoseek",
        "set_bucket_policy": "chronoseek",
    }


def test_vidaio_compressor_writes_video_response(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"raw-video")
    captured = {}

    class FakeResponse:
        headers = {"Content-Type": "video/mp4"}
        content = b"compressed-video"

        def raise_for_status(self):
            return None

    def fake_post(url, *, headers, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["filename"] = files["file"][0]
        return FakeResponse()

    monkeypatch.setattr("chronoseek.video.vidaio.requests.post", fake_post)

    result = VidaioCompressor(
        api_base_url="https://vidaio.example/api",
        api_key="token",
        timeout_seconds=12,
    ).compress(input_path=str(input_path), output_path=str(output_path))

    assert output_path.read_bytes() == b"compressed-video"
    assert result.backend == "vidaio"
    assert captured == {
        "url": "https://vidaio.example/api/compress",
        "headers": {"Authorization": "Bearer token"},
        "timeout": 12.0,
        "filename": "task.mp4",
    }


def test_composite_compressor_falls_back_to_local_compressor(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"video")

    class FailingPreferredCompressor:
        def compress(self, *, input_path: str, output_path: str):
            raise RuntimeError("remote unavailable")

    class LocalCompressor:
        def compress(self, *, input_path: str, output_path: str):
            Path(output_path).write_bytes(Path(input_path).read_bytes())
            return CompressionResult(
                path=output_path,
                profile_name="h264-720p-v1",
                backend="ffmpeg",
            )

    compressor = CompositeCompressor(
        local_compressor=LocalCompressor(),
        preferred_compressor=FailingPreferredCompressor(),
        preferred_enabled=True,
        preferred_backend_name="Preferred",
    )

    result = compressor.compress(input_path=str(input_path), output_path=str(output_path))

    assert output_path.read_bytes() == b"video"
    assert result.backend == "ffmpeg"
