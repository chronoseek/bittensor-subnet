import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from chronoseek.logging import logger
from chronoseek.validator.artifact_manifest import (
    ArtifactManifestEntry,
    TaskArtifactManifest,
)
from chronoseek.validator.artifact_publication import (
    ArtifactStorage,
    TaskArtifactPublisher,
)
from chronoseek.validator.base_task_gen import BaseTaskGenerator
from chronoseek.validator.canary_tasks import CanaryTaskPolicy
from chronoseek.validator.clipper import FfmpegClipper
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.compression import CompressionResult
from chronoseek.validator.task_models import (
    EncodingProfile,
    ValidationTask,
)
from chronoseek.validator.task_exposure import (
    ActiveTaskExposureGuard,
    TaskExposureLimits,
)
from chronoseek.validator.task_sampler import (
    ActivityNetTaskSampler,
    build_source_task_hash,
    stable_hash,
)
from chronoseek.validator.task_transforms import TaskTransformPolicy


class TaskCompressor(Protocol):
    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        ...


@dataclass(frozen=True)
class HardenedTaskGeneratorConfig:
    artifact_prefix: str = "task-clips"
    ttl_hours: float = 6.0
    cleanup_interval_seconds: float = 900.0
    max_generation_attempts: int = 5
    delete_remote_artifacts: bool = False
    enable_adversarial_transforms: bool = True
    enable_encoding_profile_variants: bool = True
    canary_task_rate: float = 0.0
    absent_canary_queries: tuple[str, ...] = (
        "a blue whale playing chess on a snow-covered tennis court",
        "a person juggling five glowing green pineapples underwater",
        "the exact phrase chronoseek absent canary appears on screen",
    )
    max_active_tasks_per_source_video: int = 0
    max_active_tasks_per_source_caption: int = 0
    max_active_tasks_per_query_variant: int = 0
    max_active_tasks_per_transform: int = 0


class HardenedActivityNetTaskGenerator(BaseTaskGenerator):
    """
    Generates private ActivityNet-derived validation tasks.

    ActivityNet remains the source of truth, but miners only receive a clipped
    task artifact URL and a query variant.
    """

    def __init__(
        self,
        *,
        sampler: ActivityNetTaskSampler,
        query_selector: QueryVariantSelector,
        clipper: FfmpegClipper,
        compressor: TaskCompressor,
        storage: ArtifactStorage,
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
        base_profile = getattr(self.clipper, "encoding_profile", None)
        if not isinstance(base_profile, EncodingProfile):
            base_profile = EncodingProfile()
        self.transforms = TaskTransformPolicy(
            base_profile=base_profile,
            enable_profile_variants=(
                self.config.enable_adversarial_transforms
                and self.config.enable_encoding_profile_variants
            ),
        )
        self.canary_policy = CanaryTaskPolicy(
            rate=self.config.canary_task_rate,
            absent_queries=self.config.absent_canary_queries,
        )
        self.exposure_guard = ActiveTaskExposureGuard(
            manifest=self.manifest,
            limits=TaskExposureLimits(
                source_video=self.config.max_active_tasks_per_source_video,
                source_caption=self.config.max_active_tasks_per_source_caption,
                query_variant=self.config.max_active_tasks_per_query_variant,
                transform=self.config.max_active_tasks_per_transform,
            ),
        )
        self.artifact_publisher = TaskArtifactPublisher(
            storage=self.storage,
            artifact_prefix=self.config.artifact_prefix,
        )
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

        delete_remote = (
            self.storage.delete_object if self.config.delete_remote_artifacts else None
        )
        deleted = self.manifest.cleanup_expired(delete_remote=delete_remote, now=now)
        self._last_cleanup_at = now
        if deleted:
            remote_mode = "with remote deletion" if delete_remote else "local only"
            logger.info(
                f"Cleaned up {deleted} expired validator task artifacts ({remote_mode})."
            )
        return deleted

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

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
                selected_profile = self.transforms.select_profile(rng=rng)
                self.transforms.apply_profile(
                    selected_profile,
                    self.clipper,
                    self.compressor,
                )
                if self.exposure_guard.is_exceeded(
                    now=now,
                    source_video_id=sample.source_video_id,
                    source_caption_id=sample.source_caption_id,
                    query_variant_id=query_variant.variant_id,
                    transform_id=selected_profile.name,
                ):
                    continue
                task_seed = stable_hash(
                    self.validator_hotkey,
                    block,
                    nonce,
                    sample.source_video_id,
                    sample.source_caption_id,
                    query_variant.variant_id,
                    selected_profile.name,
                    now,
                )
                task_id = task_seed[:24]
                hard_negative_intervals = self.transforms.hard_negative_intervals(
                    sample
                )
                clip_result = self.clipper.create_clip(
                    source_video_url=sample.source_url,
                    task_id=task_id,
                    source_video_id=sample.source_video_id,
                    ground_truths=sample.ground_truths,
                    rng=rng,
                    hard_negative_intervals=hard_negative_intervals,
                )
                clip_path = clip_result.path
                hard_negative_ids = self.transforms.hard_negative_ids(
                    sample,
                    clip_result.crop_plan,
                )
                source_task_hash = build_source_task_hash(
                    source_video_id=sample.source_video_id,
                    source_caption_id=sample.source_caption_id,
                    crop_bucket=clip_result.crop_bucket,
                    query_variant_id=query_variant.variant_id,
                    encoding_profile=selected_profile.name,
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
                artifact = self.artifact_publisher.publish(
                    task_id=task_id,
                    compression_result=compression_result,
                )

                self.manifest.add(
                    ArtifactManifestEntry(
                        task_id=task_id,
                        object_key=artifact.object_key,
                        public_url=artifact.public_url,
                        local_path=artifact.local_path,
                        source_task_hash=source_task_hash,
                        encoding_profile=compression_result.profile_name,
                        created_at=now,
                        expires_at=expires_at,
                        source_video_id=sample.source_video_id,
                        source_caption_id=sample.source_caption_id,
                        query_variant_id=query_variant.variant_id,
                        crop_bucket=clip_result.crop_bucket,
                        transform_id=selected_profile.name,
                        task_family="hardened-activitynet",
                    )
                )
                exposure = self.manifest.exposure_summary(now=now)
                logger.info(
                    "Generated hardened validator task | "
                    f"task_id={task_id} | request_id=validation-{task_id} | "
                    f"clip_duration={clip_result.crop_plan.clip_duration:.2f}s | "
                    f"query_variant={query_variant.variant_id} | "
                    f"gt_count={len(clip_result.crop_plan.shifted_ground_truths)} | "
                    f"transform={selected_profile.name} | "
                    f"hard_negatives={len(hard_negative_ids)} | "
                    f"active_artifacts={exposure['active_artifacts']}"
                )
                task = ValidationTask(
                    task_id=task_id,
                    request_id=f"validation-{task_id}",
                    video_url=artifact.public_url,
                    query=query_variant.query,
                    ground_truths=clip_result.crop_plan.shifted_ground_truths,
                    clip_duration=clip_result.crop_plan.clip_duration,
                    source_video_id=sample.source_video_id,
                    source_caption_id=sample.source_caption_id,
                    crop_start=clip_result.crop_plan.source_start,
                    crop_end=clip_result.crop_plan.source_end,
                    query_variant_id=query_variant.variant_id,
                    artifact_url=artifact.public_url,
                    artifact_key=artifact.object_key,
                    expires_at=expires_at,
                    transform_id=selected_profile.name,
                    transform_metadata=self.transforms.metadata(
                        profile=selected_profile,
                        crop_plan=clip_result.crop_plan,
                        hard_negative_count=len(hard_negative_ids),
                    ),
                    hard_negative_count=len(hard_negative_ids),
                    hard_negative_source_caption_ids=hard_negative_ids,
                )
                task = self.canary_policy.apply(task=task, rng=rng)
                manifest_entry = self.manifest.entries.get(task_id)
                if manifest_entry is not None and manifest_entry.task_family != task.task_family:
                    self.manifest.add(replace(manifest_entry, task_family=task.task_family))
                return task
            except Exception as exc:
                self._remove_local_file(clip_path)
                self._remove_local_file(compressed_path)
                last_error = exc
                logger.warning(
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
