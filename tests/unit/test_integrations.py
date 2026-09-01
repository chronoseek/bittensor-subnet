import json
from pathlib import Path

import pytest
import requests
from minio.error import S3Error

from chronoseek.hippius.s3 import (
    HippiusS3Config,
    HippiusS3ObjectRef,
    HippiusS3StorageClient,
    parse_hippius_s3_url,
)

from chronoseek.validator.compression import (
    CompositeCompressor,
    CompressionResult,
    CompressionUnavailable,
)
from chronoseek.video.vidaio import StorageBackedVidaioCompressor, VidaioCompressor


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


def test_vidaio_compressor_runs_public_workflow(monkeypatch, tmp_path):
    output_path = tmp_path / "output.mp4"
    calls = []

    class FakeResponse:
        def __init__(self, payload=None, content=b""):
            self._payload = payload or {}
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, *, headers, json, timeout):
        calls.append(("POST", url, headers, json, timeout))
        return FakeResponse(
            {
                "operation": "compression",
                "task_id": "task-123",
                "status": "queued",
                "message": "queued",
            }
        )

    def fake_get(url, *, headers=None, timeout):
        calls.append(("GET", url, headers, timeout))
        if url.endswith("/compression/task-123"):
            return FakeResponse(
                {
                    "operation": "compression",
                    "task_id": "task-123",
                    "status": "completed",
                }
            )
        if url.endswith("/compression/task-123/result"):
            return FakeResponse(
                {
                    "task_id": "task-123",
                    "download_url": "https://storage.example/output.mp4",
                }
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.post", fake_post)
    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.get", fake_get)

    result = VidaioCompressor(
        api_base_url="https://api.vidaio.io",
        api_key="token",
        timeout_seconds=12,
        poll_interval_seconds=0.25,
    ).compress_url(
        video_url="https://public.example/input.mp4",
        output_path=str(output_path),
    )

    assert result.backend == "vidaio"
    assert result.path == ""
    assert result.public_url == "https://storage.example/output.mp4"
    assert calls[0] == (
        "POST",
        "https://api.vidaio.io/compression",
        {"X-API-Key": "token"},
        {
            "video_url": "https://public.example/input.mp4",
            "compression_type": "High",
            "tier": "Standard",
        },
        12.0,
    )
    assert not output_path.exists()


def test_vidaio_compressor_reports_http_response_code(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 402

        def raise_for_status(self):
            raise requests.HTTPError("402 Payment Required")

        def json(self):
            return {"detail": "payment required"}

    def fake_post(url, *, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.post", fake_post)

    with pytest.raises(CompressionUnavailable) as exc_info:
        VidaioCompressor(
            api_base_url="https://api.vidaio.io",
            api_key="token",
            timeout_seconds=12,
        ).compress_url(
            video_url="https://public.example/input.mp4",
            output_path=str(tmp_path / "output.mp4"),
        )

    assert "HTTP 402" in str(exc_info.value)
    assert "detail=payment required" in str(exc_info.value)


def test_vidaio_compressor_backs_off_on_status_rate_limit(monkeypatch, tmp_path):
    sleeps = []
    now = [100.0]

    class FakeResponse:
        def __init__(self, payload=None, *, status_code=200, headers=None):
            self._payload = payload or {}
            self.status_code = status_code
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self._payload

    status_calls = []

    def fake_monotonic():
        return now[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def fake_post(url, *, headers, json, timeout):
        return FakeResponse(
            {
                "operation": "compression",
                "task_id": "task-123",
                "status": "queued",
                "message": "queued",
            }
        )

    def fake_get(url, *, headers=None, timeout=None):
        if url.endswith("/compression/task-123"):
            status_calls.append(url)
            if len(status_calls) == 1:
                return FakeResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            return FakeResponse(
                {
                    "operation": "compression",
                    "task_id": "task-123",
                    "status": "completed",
                }
            )
        if url.endswith("/compression/task-123/result"):
            return FakeResponse(
                {
                    "task_id": "task-123",
                    "download_url": "https://storage.example/output.mp4",
                }
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("chronoseek.video.vidaio_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("chronoseek.video.vidaio_client.time.sleep", fake_sleep)
    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.post", fake_post)
    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.get", fake_get)

    result = VidaioCompressor(
        api_base_url="https://api.vidaio.io",
        api_key="token",
        timeout_seconds=30,
        poll_interval_seconds=0.25,
    ).compress_url(
        video_url="https://public.example/input.mp4",
        output_path=str(tmp_path / "output.mp4"),
    )

    assert result.public_url == "https://storage.example/output.mp4"
    assert len(status_calls) == 2
    assert sleeps == [0.25, 1.0]


def test_vidaio_compressor_includes_failure_status(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, *, headers, json, timeout):
        return FakeResponse(
            {
                "operation": "compression",
                "task_id": "task-123",
                "status": "queued",
                "message": "queued",
            }
        )

    def fake_get(url, *, headers=None, timeout):
        return FakeResponse(
            {
                "operation": "compression",
                "task_id": "task-123",
                "status": "failed",
                "message": "source video could not be downloaded",
            }
        )

    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.post", fake_post)
    monkeypatch.setattr("chronoseek.video.vidaio_client.requests.get", fake_get)

    with pytest.raises(CompressionUnavailable) as exc_info:
        VidaioCompressor(
            api_base_url="https://api.vidaio.io",
            api_key="token",
            timeout_seconds=12,
            poll_interval_seconds=0.25,
        ).compress_url(
            video_url="https://public.example/input.mp4",
            output_path=str(tmp_path / "output.mp4"),
        )

    assert "task_id=task-123" in str(exc_info.value)
    assert "status=failed" in str(exc_info.value)
    assert "source video could not be downloaded" in str(exc_info.value)


def test_storage_backed_vidaio_compressor_uploads_temp_source(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    input_path.write_bytes(b"raw-video")

    class FakeStorage:
        def __init__(self):
            self.uploaded = []
            self.deleted = []

        def upload_file(self, *, local_path: str, object_key: str, content_type: str):
            self.uploaded.append((local_path, object_key, content_type))
            return f"https://public.example/{object_key}"

        def delete_object(self, object_key: str):
            self.deleted.append(object_key)

    captured = {}

    def fake_compress_url(self, *, video_url: str, output_path: str):
        captured["video_url"] = video_url
        return CompressionResult(
            path="",
            profile_name=self.encoding_profile.name,
            backend="vidaio",
            public_url="https://storage.example/compressed.mp4",
        )

    monkeypatch.setattr(VidaioCompressor, "compress_url", fake_compress_url)
    storage = FakeStorage()

    result = StorageBackedVidaioCompressor(
        vidaio_compressor=VidaioCompressor(
            api_base_url="https://api.vidaio.io",
            api_key=None,
            timeout_seconds=12,
        ),
        storage=storage,
        artifact_prefix="task-clips/vidaio-inputs",
    ).compress(input_path=str(input_path), output_path=str(output_path))

    assert result.backend == "vidaio"
    assert result.public_url == "https://storage.example/compressed.mp4"
    assert not output_path.exists()
    assert storage.uploaded[0][0] == str(input_path)
    assert storage.uploaded[0][1].startswith("task-clips/vidaio-inputs/")
    assert storage.deleted == [storage.uploaded[0][1]]
    assert captured["video_url"].startswith("https://public.example/task-clips/vidaio-inputs/")


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
