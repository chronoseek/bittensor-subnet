import datetime as dt
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

import requests


@dataclass(frozen=True)
class HippiusS3Config:
    endpoint_url: str
    public_base_url: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"


class HippiusS3StorageClient:
    """
    Minimal S3-compatible client for Hippius public task artifacts.

    Uses path-style S3 URLs: {endpoint}/{bucket}/{object_key}.
    """

    def __init__(self, config: HippiusS3Config, *, timeout_seconds: float = 60.0):
        self.config = config
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._validate_config()

    def _validate_config(self) -> None:
        missing = [
            name
            for name, value in {
                "endpoint_url": self.config.endpoint_url,
                "public_base_url": self.config.public_base_url,
                "bucket": self.config.bucket,
                "access_key_id": self.config.access_key_id,
                "secret_access_key": self.config.secret_access_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"Hippius S3 storage is missing required config: {', '.join(missing)}"
            )

    def _object_url(self, object_key: str) -> str:
        endpoint = self.config.endpoint_url.rstrip("/")
        bucket = quote(self.config.bucket.strip("/"), safe="")
        key = quote(object_key.strip("/"), safe="/")
        return f"{endpoint}/{bucket}/{key}"

    def public_url(self, object_key: str) -> str:
        base_url = self.config.public_base_url.rstrip("/")
        key = quote(object_key.strip("/"), safe="/")
        return f"{base_url}/{key}"

    @staticmethod
    def _signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
        def sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        date_key = sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
        region_key = sign(date_key, region)
        service_key = sign(region_key, "s3")
        return sign(service_key, "aws4_request")

    def _headers(
        self,
        *,
        method: str,
        url: str,
        body: bytes,
        content_type: str | None = None,
    ) -> dict[str, str]:
        parsed = urlparse(url)
        now = dt.datetime.now(dt.UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest()

        headers = {
            "host": parsed.netloc,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if content_type:
            headers["content-type"] = content_type

        signed_header_names = sorted(headers)
        canonical_headers = "".join(
            f"{name}:{headers[name].strip()}\n" for name in signed_header_names
        )
        signed_headers = ";".join(signed_header_names)
        canonical_uri = parsed.path or "/"
        canonical_request = "\n".join(
            [
                method.upper(),
                canonical_uri,
                parsed.query,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        credential_scope = f"{date_stamp}/{self.config.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signature = hmac.new(
            self._signing_key(
                self.config.secret_access_key,
                date_stamp,
                self.config.region,
            ),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers["authorization"] = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.config.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )
        return headers

    def upload_file(
        self,
        *,
        local_path: str,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        body = Path(local_path).read_bytes()
        url = self._object_url(object_key)
        response = requests.put(
            url,
            data=body,
            headers=self._headers(
                method="PUT",
                url=url,
                body=body,
                content_type=content_type,
            ),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self.public_url(object_key)

    def delete_object(self, object_key: str) -> None:
        url = self._object_url(object_key)
        body = b""
        response = requests.delete(
            url,
            headers=self._headers(method="DELETE", url=url, body=body),
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return
        response.raise_for_status()
