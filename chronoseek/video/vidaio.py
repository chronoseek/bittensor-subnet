from pathlib import Path

import requests

from chronoseek.validator.compression import CompressionResult, CompressionUnavailable
from chronoseek.validator.task_models import EncodingProfile


class VidaioCompressor:
    """
    Optional Vidaio adapter for validator task artifact compression.

    Vidaio is not required for validator liveness; use it behind
    CompositeCompressor so local ffmpeg remains the fallback.
    """

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        encoding_profile: EncodingProfile | None = None,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.encoding_profile = encoding_profile or EncodingProfile()

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        if not self.api_base_url:
            raise CompressionUnavailable("Vidaio compression API URL is not configured.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with open(input_path, "rb") as handle:
            response = requests.post(
                f"{self.api_base_url}/compress",
                headers=headers,
                files={"file": ("task.mp4", handle, "video/mp4")},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if content_type.startswith("video/"):
            Path(output_path).write_bytes(response.content)
        else:
            payload = response.json()
            compressed_url = payload.get("url") or payload.get("download_url")
            if not compressed_url:
                raise CompressionUnavailable(
                    "Vidaio response did not include compressed video bytes or a download URL."
                )
            download = requests.get(compressed_url, timeout=self.timeout_seconds)
            download.raise_for_status()
            Path(output_path).write_bytes(download.content)

        return CompressionResult(
            path=output_path,
            profile_name=self.encoding_profile.name,
            backend="vidaio",
        )
