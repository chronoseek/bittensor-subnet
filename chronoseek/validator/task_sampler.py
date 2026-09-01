import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

from chronoseek.validator.task_models import GroundTruthIntervals
from chronoseek.validator.video_availability import VideoAvailabilityChecker


def stable_hash(*parts: Any) -> str:
    serialized = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def seeded_rng(*parts: Any) -> random.Random:
    seed_hex = stable_hash(*parts)[:16]
    return random.Random(int(seed_hex, 16))


def build_source_caption_id(source_video_id: str, caption: str) -> str:
    return stable_hash(source_video_id, caption)[:16]


def build_source_task_hash(
    *,
    source_video_id: str,
    source_caption_id: str,
    crop_bucket: str,
    query_variant_id: str,
    encoding_profile: str,
) -> str:
    return stable_hash(
        source_video_id,
        source_caption_id,
        crop_bucket,
        query_variant_id,
        encoding_profile,
    )


@dataclass(frozen=True)
class HardNegativeMoment:
    source_caption_id: str
    caption: str
    ground_truths: GroundTruthIntervals


@dataclass(frozen=True)
class SampledActivityNetMoment:
    source_video_id: str
    source_url: str
    source_caption_id: str
    caption: str
    ground_truths: GroundTruthIntervals
    difficulty: str | None = None
    hard_negatives: tuple[HardNegativeMoment, ...] = ()


class ActivityNetTaskSampler:
    """
    Secret-seeded ActivityNet sampler with in-memory video/caption cooldowns.
    """

    def __init__(
        self,
        dataset: list[dict],
        *,
        validator_hotkey: str,
        task_secret: str,
        video_cooldown_seconds: float,
        caption_cooldown_seconds: float,
        availability_checker: VideoAvailabilityChecker | None = None,
        require_accessible_videos: bool = False,
        max_sampling_attempts: int = 100,
    ):
        if not task_secret:
            raise ValueError("VALIDATOR_TASK_SECRET is required for hardened tasks.")
        self.dataset = list(dataset)
        self.validator_hotkey = validator_hotkey
        self.task_secret = task_secret
        self.video_cooldown_seconds = max(0.0, float(video_cooldown_seconds))
        self.caption_cooldown_seconds = max(0.0, float(caption_cooldown_seconds))
        self.availability_checker = availability_checker
        self.require_accessible_videos = require_accessible_videos
        self.max_sampling_attempts = max(1, int(max_sampling_attempts))
        self._video_last_used_at: dict[str, float] = {}
        self._caption_last_used_at: dict[str, float] = {}

    def make_rng(self, *, block: int, nonce: int) -> random.Random:
        return seeded_rng(self.task_secret, self.validator_hotkey, block, nonce)

    def _cooldown_active(
        self,
        key: str,
        cache: dict[str, float],
        cooldown_seconds: float,
        now: float,
    ) -> bool:
        if cooldown_seconds <= 0:
            return False
        last_used = cache.get(key)
        if last_used is None:
            return False
        return (now - last_used) < cooldown_seconds

    @staticmethod
    def _captions_for_video(video: dict) -> list[tuple[str, GroundTruthIntervals]]:
        caption_intervals = video.get("caption_intervals") or {}
        if not isinstance(caption_intervals, dict):
            return []

        captions: list[tuple[str, GroundTruthIntervals]] = []
        for caption, intervals in caption_intervals.items():
            normalized = [(float(start), float(end)) for start, end in intervals]
            if caption and normalized:
                captions.append((str(caption), normalized))
        return captions

    def _mark_used(
        self,
        *,
        source_video_id: str,
        source_caption_id: str,
        now: float,
    ) -> None:
        self._video_last_used_at[source_video_id] = now
        self._caption_last_used_at[source_caption_id] = now

    def sample(
        self,
        *,
        block: int,
        nonce: int,
        active_source_hashes: set[str] | None = None,
    ) -> tuple[SampledActivityNetMoment, random.Random]:
        if not self.dataset:
            raise RuntimeError("ActivityNet dataset is empty.")

        rng = self.make_rng(block=block, nonce=nonce)
        active_source_hashes = active_source_hashes or set()
        now = time.time()

        candidate_indices = list(range(len(self.dataset)))
        rng.shuffle(candidate_indices)
        attempts = min(len(candidate_indices), self.max_sampling_attempts)

        for index in candidate_indices[:attempts]:
            video = self.dataset[index]
            source_url = str(video.get("video_url") or "")
            source_video_id = str(video.get("task_id") or source_url)
            if not source_url:
                continue
            if self._cooldown_active(
                source_video_id,
                self._video_last_used_at,
                self.video_cooldown_seconds,
                now,
            ):
                continue
            if self.require_accessible_videos and self.availability_checker is not None:
                availability = self.availability_checker.check(source_url)
                if not availability.accessible:
                    continue

            captions = self._captions_for_video(video)
            rng.shuffle(captions)
            for caption, ground_truths in captions:
                source_caption_id = build_source_caption_id(source_video_id, caption)
                if self._cooldown_active(
                    source_caption_id,
                    self._caption_last_used_at,
                    self.caption_cooldown_seconds,
                    now,
                ):
                    continue

                minimal_replay_key = build_source_task_hash(
                    source_video_id=source_video_id,
                    source_caption_id=source_caption_id,
                    crop_bucket="pending",
                    query_variant_id="pending",
                    encoding_profile="pending",
                )
                if minimal_replay_key in active_source_hashes:
                    continue

                self._mark_used(
                    source_video_id=source_video_id,
                    source_caption_id=source_caption_id,
                    now=now,
                )
                hard_negatives = tuple(
                    HardNegativeMoment(
                        source_caption_id=build_source_caption_id(
                            source_video_id,
                            other_caption,
                        ),
                        caption=other_caption,
                        ground_truths=other_ground_truths,
                    )
                    for other_caption, other_ground_truths in captions
                    if other_caption != caption
                )
                return (
                    SampledActivityNetMoment(
                        source_video_id=source_video_id,
                        source_url=source_url,
                        source_caption_id=source_caption_id,
                        caption=caption,
                        ground_truths=ground_truths,
                        difficulty=video.get("difficulty"),
                        hard_negatives=hard_negatives,
                    ),
                    rng,
                )

        raise RuntimeError(
            "Unable to sample a non-replayed ActivityNet task within cooldown limits."
        )
