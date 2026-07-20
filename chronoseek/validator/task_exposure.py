from dataclasses import dataclass

from chronoseek.logging import logger
from chronoseek.validator.artifact_manifest import TaskArtifactManifest


@dataclass(frozen=True)
class TaskExposureLimits:
    source_video: int = 0
    source_caption: int = 0
    query_variant: int = 0
    transform: int = 0


class ActiveTaskExposureGuard:
    """Rejects task candidates that overuse active source or transform values."""

    def __init__(self, *, manifest: TaskArtifactManifest, limits: TaskExposureLimits):
        self.manifest = manifest
        self.limits = limits

    def is_exceeded(
        self,
        *,
        now: float,
        source_video_id: str,
        source_caption_id: str,
        query_variant_id: str,
        transform_id: str,
    ) -> bool:
        checks = (
            ("source_video_id", source_video_id, self.limits.source_video),
            ("source_caption_id", source_caption_id, self.limits.source_caption),
            ("query_variant_id", query_variant_id, self.limits.query_variant),
            ("transform_id", transform_id, self.limits.transform),
        )
        for field_name, value, limit in checks:
            if not value or int(limit) <= 0:
                continue
            count = self.manifest.exposure_counts(field_name=field_name, now=now).get(
                value,
                0,
            )
            if count < int(limit):
                continue
            logger.debug(
                "Skipping hardened task candidate due to exposure limit | "
                f"field={field_name} | value={value} | count={count} | limit={limit}"
            )
            return True
        return False
