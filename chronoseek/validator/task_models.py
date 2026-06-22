from dataclasses import dataclass, field
from typing import Any

GroundTruthInterval = tuple[float, float]
GroundTruthIntervals = list[GroundTruthInterval]


@dataclass(frozen=True)
class TaskArtifact:
    url: str
    key: str
    local_path: str
    expires_at: float


@dataclass(frozen=True)
class CropPlan:
    source_start: float
    source_end: float
    clip_duration: float
    shifted_ground_truths: GroundTruthIntervals


@dataclass(frozen=True)
class EncodingProfile:
    name: str = "h264-720p-v1"
    max_width: int = 1280
    max_height: int = 720
    video_bitrate: str = "1500k"
    audio_bitrate: str = "96k"


@dataclass(frozen=True)
class HardNegativeSignal:
    source_caption_id: str
    shifted_ground_truths: GroundTruthIntervals


@dataclass(frozen=True)
class ValidationTask:
    task_id: str
    request_id: str
    video_url: str
    query: str
    ground_truths: GroundTruthIntervals
    clip_duration: float
    source_video_id: str
    source_caption_id: str
    crop_start: float
    crop_end: float
    query_variant_id: str
    artifact_url: str
    artifact_key: str
    expires_at: float
    task_family: str = "hardened-activitynet"
    transform_id: str = "base"
    transform_metadata: dict[str, Any] = field(default_factory=dict)
    hard_negative_count: int = 0
    hard_negative_source_caption_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedValidationTask:
    video_url: str
    query: str
    ground_truths: GroundTruthIntervals
    top_k: int
    clip_duration: float | None = None
    task_id: str | None = None
    request_id: str | None = None
    query_variant_id: str | None = None
    task_family: str | None = None
    transform_id: str | None = None
    hard_negative_count: int = 0


def _normalize_intervals(raw_ground_truths: Any) -> GroundTruthIntervals:
    intervals: GroundTruthIntervals = []
    for start, end in raw_ground_truths:
        intervals.append((float(start), float(end)))
    return intervals


def normalize_generated_task(
    generated_task: Any,
    *,
    default_top_k: int,
) -> NormalizedValidationTask:
    if isinstance(generated_task, ValidationTask):
        return NormalizedValidationTask(
            video_url=generated_task.video_url,
            query=generated_task.query,
            ground_truths=list(generated_task.ground_truths),
            top_k=max(1, int(default_top_k)),
            clip_duration=float(generated_task.clip_duration),
            task_id=generated_task.task_id,
            request_id=generated_task.request_id,
            query_variant_id=generated_task.query_variant_id,
            task_family=generated_task.task_family,
            transform_id=generated_task.transform_id,
            hard_negative_count=generated_task.hard_negative_count,
        )

    if isinstance(generated_task, tuple) and len(generated_task) == 3:
        video_url, query, ground_truths = generated_task
        return NormalizedValidationTask(
            video_url=str(video_url),
            query=str(query),
            ground_truths=_normalize_intervals(ground_truths),
            top_k=max(1, int(default_top_k)),
        )

    raise TypeError(
        "Task generator must return either ValidationTask or "
        "(video_url, query, ground_truths)."
    )
