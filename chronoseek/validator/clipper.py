import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.validator.task_models import CropPlan, EncodingProfile, GroundTruthIntervals


@dataclass(frozen=True)
class ClipResult:
    path: str
    crop_plan: CropPlan
    crop_bucket: str


class FfmpegClipper:
    def __init__(
        self,
        *,
        cache_dir: str,
        min_clip_duration_seconds: float = 30.0,
        max_clip_duration_seconds: float = 180.0,
        download_timeout_seconds: int = 120,
        encoding_profile: EncodingProfile | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_clip_duration_seconds = max(1.0, float(min_clip_duration_seconds))
        self.max_clip_duration_seconds = max(
            self.min_clip_duration_seconds,
            float(max_clip_duration_seconds),
        )
        self.download_timeout_seconds = max(1, int(download_timeout_seconds))
        self.encoding_profile = encoding_profile or EncodingProfile()

    @staticmethod
    def _require_binary(name: str) -> str:
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"Required binary '{name}' was not found on PATH.")
        return path

    def probe_duration(self, path: str) -> float | None:
        ffprobe = self._require_binary("ffprobe")
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            return None
        return duration if duration > 0 else None

    def build_crop_plan(
        self,
        *,
        ground_truths: GroundTruthIntervals,
        rng,
        source_duration: float | None = None,
    ) -> CropPlan:
        if not ground_truths:
            raise RuntimeError("Cannot crop a task without ground-truth intervals.")

        anchor_start, anchor_end = rng.choice(list(ground_truths))
        anchor_start = max(0.0, float(anchor_start))
        anchor_end = max(anchor_start + 0.1, float(anchor_end))
        anchor_duration = anchor_end - anchor_start

        desired_duration = max(
            self.min_clip_duration_seconds,
            min(self.max_clip_duration_seconds, anchor_duration + rng.uniform(12.0, 45.0)),
        )
        leading_context_max = max(0.0, desired_duration - anchor_duration)
        leading_context = rng.uniform(0.0, leading_context_max)
        crop_start = max(0.0, anchor_start - leading_context)
        crop_end = crop_start + desired_duration

        if source_duration is not None and crop_end > source_duration:
            crop_end = source_duration
            crop_start = max(0.0, crop_end - desired_duration)

        shifted: GroundTruthIntervals = []
        for gt_start, gt_end in ground_truths:
            overlap_start = max(float(gt_start), crop_start)
            overlap_end = min(float(gt_end), crop_end)
            if overlap_end <= overlap_start:
                continue
            shifted.append(
                (
                    round(overlap_start - crop_start, 3),
                    round(overlap_end - crop_start, 3),
                )
            )

        if not shifted:
            raise RuntimeError("Crop plan produced no clip-local ground truths.")

        clip_duration = max(0.1, crop_end - crop_start)
        return CropPlan(
            source_start=round(crop_start, 3),
            source_end=round(crop_end, 3),
            clip_duration=round(clip_duration, 3),
            shifted_ground_truths=shifted,
        )

    def _crop_with_ffmpeg(
        self,
        *,
        source_path: str,
        output_path: str,
        crop_plan: CropPlan,
    ) -> None:
        ffmpeg = self._require_binary("ffmpeg")
        profile = self.encoding_profile
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
            "-ss",
            f"{crop_plan.source_start:.3f}",
            "-i",
            source_path,
            "-t",
            f"{crop_plan.clip_duration:.3f}",
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
                f"ffmpeg clip generation failed: {result.stderr.strip() or result.stdout.strip()}"
            )

    def create_clip(
        self,
        *,
        source_video_url: str,
        task_id: str,
        source_video_id: str,
        ground_truths: GroundTruthIntervals,
        rng,
    ) -> ClipResult:
        downloaded_video = VideoDownloader.download_video(
            source_video_url,
            timeout=self.download_timeout_seconds,
        )
        if downloaded_video is None:
            raise RuntimeError("Source video could not be downloaded.")

        try:
            source_duration = self.probe_duration(downloaded_video.path)
            crop_plan = self.build_crop_plan(
                ground_truths=ground_truths,
                rng=rng,
                source_duration=source_duration,
            )
            source_dir = self.cache_dir / source_video_id.replace("/", "_")
            source_dir.mkdir(parents=True, exist_ok=True)
            output_path = source_dir / f"{task_id}.mp4"
            self._crop_with_ffmpeg(
                source_path=downloaded_video.path,
                output_path=str(output_path),
                crop_plan=crop_plan,
            )
            crop_bucket = f"{int(crop_plan.source_start // 10) * 10}-{int(crop_plan.source_end // 10) * 10}"
            return ClipResult(
                path=str(output_path),
                crop_plan=crop_plan,
                crop_bucket=crop_bucket,
            )
        finally:
            VideoDownloader.cleanup(downloaded_video)
