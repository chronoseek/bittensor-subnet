import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import bittensor as bt
import requests

from chronoseek.validator.task_models import EncodingProfile


class CompressionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CompressionResult:
    path: str
    profile_name: str
    backend: str


class LocalFfmpegCompressor:
    def __init__(self, *, encoding_profile: EncodingProfile | None = None):
        self.encoding_profile = encoding_profile or EncodingProfile()

    @staticmethod
    def _require_binary(name: str) -> str:
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"Required binary '{name}' was not found on PATH.")
        return path

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        ffmpeg = self._require_binary("ffmpeg")
        profile = self.encoding_profile
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scale_filter = (
            f"scale='min({profile.max_width},iw)':'min({profile.max_height},ih)':"
            "force_original_aspect_ratio=decrease"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-map_metadata",
            "-1",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            profile.video_bitrate,
            "-c:a",
            "aac",
            "-b:a",
            profile.audio_bitrate,
            "-movflags",
            "+faststart",
            output_path,
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg compression failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return CompressionResult(
            path=output_path,
            profile_name=profile.name,
            backend="ffmpeg",
        )


class VidaioCompressor:
    """
    Optional adapter. Vidaio is not required for validator liveness; callers should
    catch CompressionUnavailable and use LocalFfmpegCompressor as the fallback.
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


class CompositeCompressor:
    def __init__(
        self,
        *,
        local_compressor: LocalFfmpegCompressor,
        vidaio_compressor: VidaioCompressor | None = None,
        vidaio_enabled: bool = False,
    ):
        self.local_compressor = local_compressor
        self.vidaio_compressor = vidaio_compressor
        self.vidaio_enabled = bool(vidaio_enabled)

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        if self.vidaio_enabled and self.vidaio_compressor is not None:
            try:
                return self.vidaio_compressor.compress(
                    input_path=input_path,
                    output_path=output_path,
                )
            except Exception as exc:
                bt.logging.warning(
                    f"Vidaio compression failed; falling back to local ffmpeg: {exc}"
                )

        return self.local_compressor.compress(
            input_path=input_path,
            output_path=output_path,
        )
