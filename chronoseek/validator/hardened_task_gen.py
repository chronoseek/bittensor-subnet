import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import bittensor as bt

from chronoseek.validator.artifact_manifest import (
    ArtifactManifestEntry,
    TaskArtifactManifest,
)
from chronoseek.validator.base_task_gen import BaseTaskGenerator
from chronoseek.validator.clipper import FfmpegClipper
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.compression import CompressionResult
from chronoseek.validator.task_models import (
    EncodingProfile,
    GroundTruthIntervals,
    ValidationTask,
)
from chronoseek.validator.task_sampler import (
    ActivityNetTaskSampler,
    build_source_task_hash,
    stable_hash,
)


class TaskCompressor(Protocol):
    def compress(self, *, input_path: str, output_path: str) -> CompressionResult:
        ...


class TaskArtifactStorage(Protocol):
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
        storage: TaskArtifactStorage,
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

        delete_remote = (
            self.storage.delete_object if self.config.delete_remote_artifacts else None
        )
        deleted = self.manifest.cleanup_expired(delete_remote=delete_remote, now=now)
        self._last_cleanup_at = now
        if deleted:
            remote_mode = "with remote deletion" if delete_remote else "local only"
            bt.logging.info(
                f"Cleaned up {deleted} expired validator task artifacts ({remote_mode})."
            )
        return deleted

    def _next_nonce(self) -> int:
        self._nonce += 1
        return self._nonce

    def _object_key(self, *, task_id: str, created_at: float) -> str:
        prefix = self.config.artifact_prefix.strip("/")
        return f"{prefix}/{task_id}.mp4"

    @staticmethod
    def _scale_bitrate(value: str, factor: float) -> str:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", str(value).strip())
        if not match:
            return value
        number = float(match.group(1))
        suffix = match.group(2)
        scaled = max(1, int(round(number * float(factor))))
        return f"{scaled}{suffix}"

    @staticmethod
    def _profile_variants(base_profile: EncodingProfile) -> list[EncodingProfile]:
        return [
            EncodingProfile(
                name=f"{base_profile.name}:base",
                max_width=base_profile.max_width,
                max_height=base_profile.max_height,
                video_bitrate=base_profile.video_bitrate,
                audio_bitrate=base_profile.audio_bitrate,
            ),
            EncodingProfile(
                name=f"{base_profile.name}:compact",
                max_width=min(base_profile.max_width, 960),
                max_height=min(base_profile.max_height, 540),
                video_bitrate=HardenedActivityNetTaskGenerator._scale_bitrate(
                    base_profile.video_bitrate,
                    0.80,
                ),
                audio_bitrate=base_profile.audio_bitrate,
            ),
            EncodingProfile(
                name=f"{base_profile.name}:detail",
                max_width=base_profile.max_width,
                max_height=base_profile.max_height,
                video_bitrate=HardenedActivityNetTaskGenerator._scale_bitrate(
                    base_profile.video_bitrate,
                    1.15,
                ),
                audio_bitrate=base_profile.audio_bitrate,
            ),
        ]

    @staticmethod
    def _encoding_profile(component: Any) -> EncodingProfile | None:
        profile = getattr(component, "encoding_profile", None)
        return profile if isinstance(profile, EncodingProfile) else None

    def _select_encoding_profile(self, *, rng) -> EncodingProfile:
        base_profile = self._encoding_profile(self.clipper) or EncodingProfile()
        if (
            not self.config.enable_adversarial_transforms
            or not self.config.enable_encoding_profile_variants
        ):
            return base_profile
        return rng.choice(self._profile_variants(base_profile))

    @staticmethod
    def _apply_encoding_profile(component: Any, profile: EncodingProfile) -> None:
        if component is None:
            return
        if hasattr(component, "encoding_profile"):
            try:
                setattr(component, "encoding_profile", profile)
            except Exception:
                pass
        for child_name in ("local_compressor", "preferred_compressor"):
            child = getattr(component, child_name, None)
            if child is not None:
                HardenedActivityNetTaskGenerator._apply_encoding_profile(child, profile)

    @staticmethod
    def _flatten_intervals(groups: list[GroundTruthIntervals]) -> GroundTruthIntervals:
        intervals: GroundTruthIntervals = []
        for group in groups:
            intervals.extend(group)
        return intervals

    @staticmethod
    def _overlap_seconds(left: tuple[float, float], right: tuple[float, float]) -> float:
        return max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))

    @classmethod
    def _hard_negative_ids_in_crop(cls, sample, crop_plan) -> tuple[str, ...]:
        crop_interval = (float(crop_plan.source_start), float(crop_plan.source_end))
        ids: list[str] = []
        for negative in getattr(sample, "hard_negatives", ()):
            if any(
                cls._overlap_seconds((float(start), float(end)), crop_interval) > 0.0
                for start, end in negative.ground_truths
            ):
                ids.append(negative.source_caption_id)
        return tuple(sorted(set(ids)))

    @staticmethod
    def _transform_metadata(
        *,
        profile: EncodingProfile,
        clip_result,
        hard_negative_count: int,
    ) -> dict[str, Any]:
        crop_plan = clip_result.crop_plan
        first_gt_start = min(start for start, _ in crop_plan.shifted_ground_truths)
        last_gt_end = max(end for _, end in crop_plan.shifted_ground_truths)
        return {
            "encoding_profile": profile.name,
            "max_width": profile.max_width,
            "max_height": profile.max_height,
            "video_bitrate": profile.video_bitrate,
            "audio_bitrate": profile.audio_bitrate,
            "leading_context_seconds": round(float(first_gt_start), 3),
            "trailing_context_seconds": round(
                max(0.0, float(crop_plan.clip_duration) - float(last_gt_end)),
                3,
            ),
            "hard_negative_count": int(hard_negative_count),
        }

    def _maybe_apply_canary(self, *, task: ValidationTask, sample, rng) -> ValidationTask:
        del sample
        rate = max(0.0, min(1.0, float(self.config.canary_task_rate)))
        if rate <= 0.0 or rng.random() >= rate:
            return task

        candidates = ["absent"]
        if task.hard_negative_count > 0:
            candidates.append("hard-negative")
        if len(task.ground_truths) > 1:
            candidates.append("repeated")

        canary_kind = rng.choice(candidates)
        metadata = {
            **task.transform_metadata,
            "canary_kind": canary_kind,
            "canary_source": "validator",
        }

        if canary_kind == "absent":
            absent_queries = tuple(
                query.strip()
                for query in self.config.absent_canary_queries
                if query.strip()
            )
            query = rng.choice(absent_queries or ("an event that is not present",))
            return replace(
                task,
                task_family="canary-absent",
                query=query,
                ground_truths=[],
                canary_kind=canary_kind,
                expects_empty_response=True,
                transform_metadata=metadata,
            )

        return replace(
            task,
            task_family=f"canary-{canary_kind}",
            canary_kind=canary_kind,
            expects_empty_response=False,
            transform_metadata=metadata,
        )

    @staticmethod
    def _limit_exceeded(*, current_count: int, limit: int) -> bool:
        return int(limit) > 0 and int(current_count) >= int(limit)

    def _exposure_limit_exceeded(
        self,
        *,
        now: float,
        source_video_id: str,
        source_caption_id: str,
        query_variant_id: str,
        transform_id: str,
    ) -> bool:
        checks = [
            (
                "source_video_id",
                source_video_id,
                self.config.max_active_tasks_per_source_video,
            ),
            (
                "source_caption_id",
                source_caption_id,
                self.config.max_active_tasks_per_source_caption,
            ),
            (
                "query_variant_id",
                query_variant_id,
                self.config.max_active_tasks_per_query_variant,
            ),
            (
                "transform_id",
                transform_id,
                self.config.max_active_tasks_per_transform,
            ),
        ]
        for field_name, value, limit in checks:
            if not value or int(limit) <= 0:
                continue
            count = self.manifest.exposure_counts(field_name=field_name, now=now).get(
                value,
                0,
            )
            if self._limit_exceeded(current_count=count, limit=limit):
                bt.logging.debug(
                    "Skipping hardened task candidate due to exposure limit | "
                    f"field={field_name} | value={value} | count={count} | limit={limit}"
                )
                return True
        return False

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
                selected_profile = self._select_encoding_profile(rng=rng)
                self._apply_encoding_profile(self.clipper, selected_profile)
                self._apply_encoding_profile(self.compressor, selected_profile)
                if self._exposure_limit_exceeded(
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
                hard_negative_intervals = self._flatten_intervals(
                    [
                        negative.ground_truths
                        for negative in getattr(sample, "hard_negatives", ())
                    ]
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
                hard_negative_ids = self._hard_negative_ids_in_crop(
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
                        source_video_id=sample.source_video_id,
                        source_caption_id=sample.source_caption_id,
                        query_variant_id=query_variant.variant_id,
                        crop_bucket=clip_result.crop_bucket,
                        transform_id=selected_profile.name,
                        task_family="hardened-activitynet",
                    )
                )
                exposure = self.manifest.exposure_summary(now=now)
                bt.logging.info(
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
                    transform_id=selected_profile.name,
                    transform_metadata=self._transform_metadata(
                        profile=selected_profile,
                        clip_result=clip_result,
                        hard_negative_count=len(hard_negative_ids),
                    ),
                    hard_negative_count=len(hard_negative_ids),
                    hard_negative_source_caption_ids=hard_negative_ids,
                )
                task = self._maybe_apply_canary(task=task, sample=sample, rng=rng)
                manifest_entry = self.manifest.entries.get(task_id)
                if manifest_entry is not None and manifest_entry.task_family != task.task_family:
                    self.manifest.add(replace(manifest_entry, task_family=task.task_family))
                return task
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
