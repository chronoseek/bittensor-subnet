import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import bittensor as bt

from chronoseek.validator.task_models import EncodingProfile


class CompressionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CompressionResult:
    path: str
    profile_name: str
    backend: str
    public_url: str = ""


class VideoCompressor(Protocol):
    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        ...


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


class CompositeCompressor:
    def __init__(
        self,
        *,
        local_compressor: VideoCompressor,
        preferred_compressor: VideoCompressor | None = None,
        preferred_enabled: bool = False,
        preferred_backend_name: str = "remote",
    ):
        self.local_compressor = local_compressor
        self.preferred_compressor = preferred_compressor
        self.preferred_enabled = bool(preferred_enabled)
        self.preferred_backend_name = preferred_backend_name or "remote"

    @property
    def encoding_profile(self) -> EncodingProfile:
        profile = getattr(self.local_compressor, "encoding_profile", None)
        return profile if isinstance(profile, EncodingProfile) else EncodingProfile()

    @encoding_profile.setter
    def encoding_profile(self, profile: EncodingProfile) -> None:
        for compressor in (self.local_compressor, self.preferred_compressor):
            if compressor is not None and hasattr(compressor, "encoding_profile"):
                compressor.encoding_profile = profile

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        if self.preferred_enabled and self.preferred_compressor is not None:
            try:
                return self.preferred_compressor.compress(
                    input_path=input_path,
                    output_path=output_path,
                )
            except Exception as exc:
                bt.logging.warning(
                    f"{self.preferred_backend_name} compression failed; falling back to local ffmpeg: {exc}"
                )

        return self.local_compressor.compress(
            input_path=input_path,
            output_path=output_path,
        )
