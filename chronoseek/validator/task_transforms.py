import re
from typing import Any

from chronoseek.validator.task_models import (
    CropPlan,
    EncodingProfile,
    GroundTruthIntervals,
)


def _scale_bitrate(value: str, factor: float) -> str:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", str(value).strip())
    if not match:
        return value
    number = float(match.group(1))
    suffix = match.group(2)
    scaled = max(1, int(round(number * float(factor))))
    return f"{scaled}{suffix}"


def encoding_profile_variants(base: EncodingProfile) -> tuple[EncodingProfile, ...]:
    return (
        EncodingProfile(
            name=f"{base.name}:base",
            max_width=base.max_width,
            max_height=base.max_height,
            video_bitrate=base.video_bitrate,
            audio_bitrate=base.audio_bitrate,
        ),
        EncodingProfile(
            name=f"{base.name}:compact",
            max_width=min(base.max_width, 960),
            max_height=min(base.max_height, 540),
            video_bitrate=_scale_bitrate(base.video_bitrate, 0.80),
            audio_bitrate=base.audio_bitrate,
        ),
        EncodingProfile(
            name=f"{base.name}:detail",
            max_width=base.max_width,
            max_height=base.max_height,
            video_bitrate=_scale_bitrate(base.video_bitrate, 1.15),
            audio_bitrate=base.audio_bitrate,
        ),
    )


class TaskTransformPolicy:
    """Selects and applies per-task encoding transforms from a stable base."""

    def __init__(
        self,
        *,
        base_profile: EncodingProfile,
        enable_profile_variants: bool,
    ):
        self.base_profile = base_profile
        self.enable_profile_variants = bool(enable_profile_variants)
        self.profile_variants = encoding_profile_variants(base_profile)

    def select_profile(self, *, rng) -> EncodingProfile:
        if not self.enable_profile_variants:
            return self.base_profile
        return rng.choice(self.profile_variants)

    @staticmethod
    def apply_profile(profile: EncodingProfile, *components: Any) -> None:
        for component in components:
            if component is not None and hasattr(component, "encoding_profile"):
                component.encoding_profile = profile

    @staticmethod
    def hard_negative_intervals(sample) -> GroundTruthIntervals:
        intervals: GroundTruthIntervals = []
        for negative in getattr(sample, "hard_negatives", ()):
            intervals.extend(negative.ground_truths)
        return intervals

    @staticmethod
    def hard_negative_ids(sample, crop_plan: CropPlan) -> tuple[str, ...]:
        crop_start = float(crop_plan.source_start)
        crop_end = float(crop_plan.source_end)
        ids = {
            negative.source_caption_id
            for negative in getattr(sample, "hard_negatives", ())
            if any(
                min(float(end), crop_end) - max(float(start), crop_start) > 0.0
                for start, end in negative.ground_truths
            )
        }
        return tuple(sorted(ids))

    @staticmethod
    def metadata(
        *,
        profile: EncodingProfile,
        crop_plan: CropPlan,
        hard_negative_count: int,
    ) -> dict[str, Any]:
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
