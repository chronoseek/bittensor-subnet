from dataclasses import dataclass
from typing import Protocol

from chronoseek.validator.compression import CompressionResult


class ArtifactStorage(Protocol):
    def upload_file(
        self,
        *,
        local_path: str,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        ...

    def delete_object(self, object_key: str) -> None:
        ...


@dataclass(frozen=True)
class PublishedArtifact:
    public_url: str
    object_key: str
    local_path: str


class TaskArtifactPublisher:
    """Publishes local compression results and accepts already-hosted results."""

    def __init__(self, *, storage: ArtifactStorage, artifact_prefix: str):
        self.storage = storage
        self.artifact_prefix = artifact_prefix.strip("/") or "task-clips"

    def object_key(self, task_id: str) -> str:
        return f"{self.artifact_prefix}/{task_id}.mp4"

    def publish(
        self,
        *,
        task_id: str,
        compression_result: CompressionResult,
    ) -> PublishedArtifact:
        if compression_result.public_url:
            return PublishedArtifact(
                public_url=compression_result.public_url,
                object_key="",
                local_path=compression_result.path,
            )

        if not compression_result.path:
            raise RuntimeError("Compressor did not return a local path or public URL.")

        object_key = self.object_key(task_id)
        public_url = self.storage.upload_file(
            local_path=compression_result.path,
            object_key=object_key,
            content_type="video/mp4",
        )
        return PublishedArtifact(
            public_url=public_url,
            object_key=object_key,
            local_path=compression_result.path,
        )
