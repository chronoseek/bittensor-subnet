import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, unquote, urlparse

from minio import Minio
from minio.error import S3Error
import requests
import urllib3

from chronoseek.constants import (
    DEFAULT_HIPPIUS_S3_BUCKET,
    DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
    DEFAULT_HIPPIUS_S3_PUBLIC_BASE_URL,
    DEFAULT_HIPPIUS_S3_REGION,
)

HIPPIUS_S3_HOSTS = {
    "s3.hippius.com",
    "eu-central-1.hippius.com",
    "us-east-1.hippius.com",
}


@dataclass(frozen=True)
class HippiusS3Config:
    endpoint_url: str = DEFAULT_HIPPIUS_S3_ENDPOINT_URL
    public_base_url: str = DEFAULT_HIPPIUS_S3_PUBLIC_BASE_URL
    bucket: str = DEFAULT_HIPPIUS_S3_BUCKET
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = DEFAULT_HIPPIUS_S3_REGION
    anonymous: bool = False


@dataclass(frozen=True)
class HippiusS3ObjectRef:
    bucket: str
    object_key: str


def normalize_endpoint_url(endpoint_url: str) -> str:
    value = (endpoint_url or DEFAULT_HIPPIUS_S3_ENDPOINT_URL).strip().rstrip("/")
    if not value:
        value = DEFAULT_HIPPIUS_S3_ENDPOINT_URL
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def _host(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    parseable = raw if "://" in raw or raw.startswith("//") else f"//{raw}"
    parsed = urlparse(parseable)
    return (parsed.hostname or "").lower()


def is_hippius_s3_url(url: str) -> bool:
    host = _host(url)
    return host in HIPPIUS_S3_HOSTS or host.endswith(".hippius.com")


def parse_hippius_s3_url(
    url: str,
    *,
    default_bucket: str | None = None,
    public_base_url: str | None = None,
) -> HippiusS3ObjectRef | None:
    """
    Parse a Hippius public S3 URL into bucket/key components.

    Supports documented path-style URLs:
    https://s3.hippius.com/{bucket}/{object-key}

    For custom public bases mapped to a bucket, pass default_bucket.
    """
    host_is_hippius = is_hippius_s3_url(url)
    if not host_is_hippius and not public_base_url:
        return None

    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if not path:
        return None

    default_bucket = (default_bucket or "").strip("/")
    base_matched = False
    base_bucket = ""
    if public_base_url:
        public_base_raw = public_base_url.strip()
        parseable_public_base = (
            public_base_raw
            if "://" in public_base_raw or public_base_raw.startswith("//")
            else f"//{public_base_raw}"
        )
        public_base = urlparse(parseable_public_base)
        if _host(public_base_url) == _host(url):
            base_matched = True
            base_path = (public_base.path or "").strip("/")
            base_path_parts = [part for part in base_path.split("/") if part]
            if _host(public_base_url) in HIPPIUS_S3_HOSTS and base_path_parts:
                base_bucket = unquote(base_path_parts[0])
            if base_path and path.startswith(f"{base_path}/"):
                path = path[len(base_path) + 1 :]
            elif base_path and path == base_path:
                path = ""
            elif base_path:
                base_matched = False
        elif not host_is_hippius:
            return None

    if base_matched and base_bucket:
        bucket = base_bucket
        object_key = path
    elif host_is_hippius:
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        bucket, object_key = parts
    elif base_matched and default_bucket:
        bucket = default_bucket
        object_key = path
    else:
        return None

    bucket = unquote(bucket).strip("/")
    object_key = unquote(object_key).strip("/")
    if not bucket or not object_key:
        return None
    return HippiusS3ObjectRef(bucket=bucket, object_key=object_key)


def stream_public_object(
    *,
    url: str,
    timeout_seconds: float,
) -> Iterator[bytes]:
    response = requests.get(url, stream=True, timeout=max(1.0, float(timeout_seconds)))
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            yield chunk


def download_public_file(
    *,
    url: str,
    local_path: str,
    timeout_seconds: float = 60.0,
) -> str:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as handle:
        for chunk in stream_public_object(
            url=url,
            timeout_seconds=timeout_seconds,
        ):
            handle.write(chunk)
    return local_path


class HippiusS3StorageClient:
    """
    Minimal S3-compatible client for Hippius public task artifacts.

    Uses the MinIO S3 SDK with path-style public URLs:
    {endpoint}/{bucket}/{object_key}.
    """

    def __init__(self, config: HippiusS3Config, *, timeout_seconds: float = 60.0):
        self.config = config
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._validate_config()
        self._client = self._build_client()

    def _validate_config(self) -> None:
        missing = [
            name
            for name, value in {
                "endpoint_url": self.config.endpoint_url,
                "bucket": self.config.bucket,
            }.items()
            if not value
        ]
        if not self.config.anonymous:
            missing.extend(
                name
                for name, value in {
                    "access_key_id": self.config.access_key_id,
                    "secret_access_key": self.config.secret_access_key,
                }.items()
                if not value
            )
        if missing:
            raise ValueError(
                f"Hippius S3 storage is missing required config: {', '.join(missing)}"
            )

    def _require_credentials(self, operation: str) -> None:
        if self.config.access_key_id and self.config.secret_access_key:
            return
        raise ValueError(
            f"Hippius S3 {operation} requires access_key_id and secret_access_key."
        )

    def _build_client(self) -> Minio:
        endpoint_url = normalize_endpoint_url(self.config.endpoint_url)
        parsed = urlparse(endpoint_url)
        endpoint = parsed.netloc
        if not endpoint:
            raise ValueError("Hippius S3 storage endpoint_url is invalid.")
        if parsed.path and parsed.path != "/":
            raise ValueError("Hippius S3 endpoint_url must not include a path.")

        timeout = urllib3.Timeout(
            connect=self.timeout_seconds,
            read=self.timeout_seconds,
        )
        http_client = urllib3.PoolManager(timeout=timeout)
        return Minio(
            endpoint,
            access_key=self.config.access_key_id or None,
            secret_key=self.config.secret_access_key or None,
            secure=parsed.scheme != "http",
            region=self.config.region,
            http_client=http_client,
        )

    def _object_url(self, object_key: str) -> str:
        endpoint = normalize_endpoint_url(self.config.endpoint_url)
        bucket = quote(self.config.bucket.strip("/"), safe="")
        key = quote(object_key.strip("/"), safe="/")
        return f"{endpoint}/{bucket}/{key}"

    def public_read_bucket_policy(self) -> str:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self.config.bucket}/*"],
                }
            ],
        }
        return json.dumps(policy, sort_keys=True)

    def ensure_bucket_exists(self) -> None:
        self._require_credentials("bucket setup")
        if self._client.bucket_exists(self.config.bucket):
            return
        self._client.make_bucket(self.config.bucket)

    def ensure_bucket_public(self) -> None:
        """
        Ensure the configured Hippius bucket exists and permits public object reads.

        Hippius buckets are private by default. Validators publish task clips as
        public objects, so startup applies the documented public-read bucket
        policy before any hardened task uploads.
        """
        self.ensure_bucket_exists()
        try:
            self._client.set_bucket_policy(
                self.config.bucket,
                self.public_read_bucket_policy(),
            )
        except S3Error as exc:
            if exc.code == "PolicyAlreadyExists":
                return
            raise

    def public_url(self, object_key: str) -> str:
        base_url = (self.config.public_base_url or self.config.endpoint_url).rstrip("/")
        if not base_url:
            base_url = DEFAULT_HIPPIUS_S3_ENDPOINT_URL
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"

        parsed = urlparse(base_url)
        bucket = quote(self.config.bucket.strip("/"), safe="")
        key = quote(object_key.strip("/"), safe="/")
        base_path_parts = [part for part in (parsed.path or "").split("/") if part]
        if _host(base_url) in HIPPIUS_S3_HOSTS and (
            not base_path_parts or base_path_parts[0] != self.config.bucket
        ):
            return f"{base_url}/{bucket}/{key}"
        return f"{base_url}/{key}"

    def upload_file(
        self,
        *,
        local_path: str,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        self._require_credentials("upload")
        self._client.fput_object(
            self.config.bucket,
            object_key.strip("/"),
            local_path,
            content_type=content_type,
        )
        return self.public_url(object_key)

    def download_file(
        self,
        *,
        object_key: str,
        local_path: str,
    ) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(
            self.config.bucket,
            object_key.strip("/"),
            local_path,
        )
        return local_path

    def download_public_url(
        self,
        *,
        url: str,
        local_path: str,
    ) -> str:
        return download_public_file(
            url=url,
            local_path=local_path,
            timeout_seconds=self.timeout_seconds,
        )

    def delete_object(self, object_key: str) -> None:
        self._require_credentials("delete")
        try:
            self._client.remove_object(self.config.bucket, object_key.strip("/"))
        except S3Error as exc:
            if exc.code in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
                return
            raise
