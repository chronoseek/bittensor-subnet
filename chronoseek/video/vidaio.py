import uuid

from chronoseek.validator.artifact_publication import ArtifactStorage
from chronoseek.validator.compression import CompressionResult, CompressionUnavailable
from chronoseek.validator.task_models import EncodingProfile
from chronoseek.video.vidaio_client import VidaioApiClient


class VidaioCompressor:
    """
    Vidaio adapter for validator task artifact compression.

    The public Vidaio API accepts a source video URL and returns an async task.
    Use it behind CompositeCompressor so local ffmpeg remains the fallback.
    """

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        compression_type: str = "High",
        tier: str = "Standard",
        poll_interval_seconds: float = 15.0,
        encoding_profile: EncodingProfile | None = None,
    ):
        self.client = VidaioApiClient(
            api_base_url=api_base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            compression_type=compression_type,
            tier=tier,
            poll_interval_seconds=poll_interval_seconds,
        )
        self.encoding_profile = encoding_profile or EncodingProfile()

    def compress_url(self, *, video_url: str, output_path: str) -> CompressionResult:
        del output_path
        result_url = self.client.compress(video_url=video_url)

        return CompressionResult(
            path="",
            profile_name=self.encoding_profile.name,
            backend="vidaio",
            public_url=result_url,
        )

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        if input_path.startswith(("http://", "https://")):
            return self.compress_url(video_url=input_path, output_path=output_path)
        raise CompressionUnavailable(
            "Vidaio compression requires a public source URL; use "
            "StorageBackedVidaioCompressor for local files."
        )


class StorageBackedVidaioCompressor:
    """
    Uploads a local clip to temporary public storage, compresses that URL through
    Vidaio, then returns Vidaio's hosted compressed result URL.
    """

    def __init__(
        self,
        *,
        vidaio_compressor: VidaioCompressor,
        storage: ArtifactStorage,
        artifact_prefix: str = "task-clips/vidaio-inputs",
        delete_remote_artifacts: bool = True,
    ):
        self.vidaio_compressor = vidaio_compressor
        self.storage = storage
        self.artifact_prefix = artifact_prefix.strip("/") or "task-clips/vidaio-inputs"
        self.delete_remote_artifacts = bool(delete_remote_artifacts)

    @property
    def encoding_profile(self) -> EncodingProfile:
        return self.vidaio_compressor.encoding_profile

    @encoding_profile.setter
    def encoding_profile(self, profile: EncodingProfile) -> None:
        self.vidaio_compressor.encoding_profile = profile

    def _object_key(self) -> str:
        return f"{self.artifact_prefix}/{uuid.uuid4().hex}.mp4"

    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        object_key = self._object_key()
        source_url = self.storage.upload_file(
            local_path=input_path,
            object_key=object_key,
            content_type="video/mp4",
        )
        try:
            return self.vidaio_compressor.compress_url(
                video_url=source_url,
                output_path=output_path,
            )
        finally:
            if self.delete_remote_artifacts:
                try:
                    self.storage.delete_object(object_key)
                except Exception:
                    pass
