import os
import time
from dataclasses import dataclass
from pathlib import Path

import bittensor as bt

from chronoseek.validator.artifact_manifest import (
    ArtifactManifestEntry,
    TaskArtifactManifest,
)
from chronoseek.validator.base_task_gen import BaseTaskGenerator
from chronoseek.validator.clipper import FfmpegClipper
from chronoseek.validator.compression import CompositeCompressor
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.storage import HippiusS3StorageClient
from chronoseek.validator.task_models import ValidationTask
from chronoseek.validator.task_sampler import (
    ActivityNetTaskSampler,
    build_source_task_hash,
    stable_hash,
)


@dataclass(frozen=True)
class HardenedTaskGeneratorConfig:
    artifact_prefix: str = "validator-tasks"
    ttl_hours: float = 6.0
    cleanup_interval_seconds: float = 900.0
    max_generation_attempts: int = 5


class HardenedActivityNetTaskGenerator(BaseTaskGenerator):
    """
    Generates private ActivityNet-derived validation tasks.

    ActivityNet remains the source of truth, but miners only receive a clipped
    Hippius artifact URL and a query variant.
    """

    def __init__(
        self,
        *,
        sampler: ActivityNetTaskSampler,
        query_selector: QueryVariantSelector,
        clipper: FfmpegClipper,
        compressor: CompositeCompressor,
        storage: HippiusS3StorageClient,
        manifest: TaskArtifactManifest,
        validator_hotkey: str,
        current_block_provider,
        config: HardenedTaskGeneratorConfig | None = None,
    ):
        self.sampler = sampler
        self.query_selector = query_selector
        self.clipper = clipper
        self.compressor = compressor
        self.storage = storage
        self.manifest = manifest
        self.validator_hotkey = validator_hotkey
        self.current_block_provider = current_block_provider
        self.config = config or HardenedTaskGeneratorConfig()
        self._nonce = 0
        self._last_cleanup_at = 0.0

    def cleanup_expired(self, *, force: bool = False) -> int:
        now = time.time()
        if (
            not force
            and now - self._last_cleanup_at
            < max(1.0, self.config.cleanup_interval_seconds)
        ):
            return 0

        deleted = self.manifest.cleanup_expired(
            delete_remote=self.storage.delete_object,
            now=now,
        )
        self._last_cleanup_at = now
        if deleted:
            bt.logging.info(f"Cleaned up {deleted} expired validator task artifacts.")
        return deleted

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def _object_key(self, *, task_id: str, created_at: float) -> str:
        day = time.strftime("%Y-%m-%d", time.gmtime(created_at))
        hotkey_prefix = "".join(
            char for char in self.validator_hotkey if char.isalnum()
        )[:16]
        prefix = self.config.artifact_prefix.strip("/")
        return f"{prefix}/{hotkey_prefix}/{day}/{task_id}.mp4"

    @staticmethod
    def _remove_local_file(path: str | None) -> None:
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def generate_task(self) -> ValidationTask:
        self.cleanup_expired()
        now = time.time()
        expires_at = now + (float(self.config.ttl_hours) * 3600.0)
        block = int(self.current_block_provider())
        active_source_hashes = self.manifest.active_source_hashes(now=now)

        last_error: Exception | None = None
        for _ in range(max(1, int(self.config.max_generation_attempts))):
            nonce = self._next_nonce()
            clip_path: str | None = None
            compressed_path: str | None = None
            try:
                sample, rng = self.sampler.sample(
                    block=block,
                    nonce=nonce,
                    active_source_hashes=active_source_hashes,
                )
                query_variant = self.query_selector.select(
                    caption=sample.caption,
                    source_caption_id=sample.source_caption_id,
                    rng=rng,
                )
                task_seed = stable_hash(
                    self.validator_hotkey,
                    block,
                    nonce,
                    sample.source_video_id,
                    sample.source_caption_id,
                    query_variant.variant_id,
                    now,
                )
                task_id = task_seed[:24]
                clip_result = self.clipper.create_clip(
                    source_video_url=sample.source_url,
                    task_id=task_id,
                    source_video_id=sample.source_video_id,
                    ground_truths=sample.ground_truths,
                    rng=rng,
                )
                clip_path = clip_result.path
                source_task_hash = build_source_task_hash(
                    source_video_id=sample.source_video_id,
                    source_caption_id=sample.source_caption_id,
                    crop_bucket=clip_result.crop_bucket,
                    query_variant_id=query_variant.variant_id,
                    encoding_profile=self.clipper.encoding_profile.name,
                )
                if source_task_hash in active_source_hashes:
                    self._remove_local_file(clip_path)
                    clip_path = None
                    continue

                compressed_path = str(
                    Path(clip_result.path).with_name(f"{task_id}.compressed.mp4")
                )
                compression_result = self.compressor.compress(
                    input_path=clip_result.path,
                    output_path=compressed_path,
                )
                if compression_result.path != clip_result.path:
                    self._remove_local_file(clip_result.path)
                    clip_path = None
                object_key = self._object_key(task_id=task_id, created_at=now)
                artifact_url = self.storage.upload_file(
                    local_path=compression_result.path,
                    object_key=object_key,
                    content_type="video/mp4",
                )

                self.manifest.add(
                    ArtifactManifestEntry(
                        task_id=task_id,
                        object_key=object_key,
                        public_url=artifact_url,
                        local_path=compression_result.path,
                        source_task_hash=source_task_hash,
                        encoding_profile=compression_result.profile_name,
                        created_at=now,
                        expires_at=expires_at,
                    )
                )
                bt.logging.info(
                    "Generated hardened validator task | "
                    f"task_id={task_id} | request_id=validation-{task_id} | "
                    f"clip_duration={clip_result.crop_plan.clip_duration:.2f}s | "
                    f"query_variant={query_variant.variant_id} | "
                    f"gt_count={len(clip_result.crop_plan.shifted_ground_truths)}"
                )
                return ValidationTask(
                    task_id=task_id,
                    request_id=f"validation-{task_id}",
                    video_url=artifact_url,
                    query=query_variant.query,
                    ground_truths=clip_result.crop_plan.shifted_ground_truths,
                    clip_duration=clip_result.crop_plan.clip_duration,
                    source_video_id=sample.source_video_id,
                    source_caption_id=sample.source_caption_id,
                    crop_start=clip_result.crop_plan.source_start,
                    crop_end=clip_result.crop_plan.source_end,
                    query_variant_id=query_variant.variant_id,
                    artifact_url=artifact_url,
                    artifact_key=object_key,
                    expires_at=expires_at,
                )
            except Exception as exc:
                self._remove_local_file(clip_path)
                self._remove_local_file(compressed_path)
                last_error = exc
                bt.logging.warning(
                    f"Hardened validator task generation attempt failed: {exc}"
                )

        raise RuntimeError(
            f"Unable to generate hardened validator task after retries: {last_error}"
        )

    @staticmethod
    def ensure_private_asset_path(path: str | None, *, description: str) -> str | None:
        if not path:
            return None
        expanded = str(Path(path).expanduser())
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"{description} was not found: {expanded}")
        return expanded
