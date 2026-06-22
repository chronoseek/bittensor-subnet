#!/usr/bin/env python3
"""Generate and inspect one hardened validator task without miner fanout."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()


DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "chronoseek" / "hardened-task-verifier")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def iso_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


class LocalDryRunStorage:
    """Storage shim that copies uploads locally and returns synthetic HTTPS URLs."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.uploads: dict[str, str] = {}
        self.public_base_url = "https://local-hardened-task.invalid"

    def upload_file(
        self,
        *,
        local_path: str,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        del content_type
        destination = self.root / object_key.strip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        self.uploads[object_key] = str(destination)
        return f"{self.public_base_url}/{object_key.strip('/')}"

    def delete_object(self, object_key: str) -> None:
        path = self.root / object_key.strip("/")
        path.unlink(missing_ok=True)


class RecordingTaskSampler:
    """Sampler wrapper that records the source task selected by the generator."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.last_sample = None

    def sample(self, **kwargs):
        sample, rng = self.sampler.sample(**kwargs)
        self.last_sample = sample
        return sample, rng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one hardened ActivityNet-derived validator task and print the "
            "task/request metadata without querying miners."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("TASK_DATASET_PATH", ""),
        help=(
            "Optional local ActivityNet-style manifest. If omitted, the script uses "
            "the Hugging Face ActivityNet loader and requires HF_TOKEN."
        ),
    )
    parser.add_argument(
        "--use-smoke-dataset",
        action="store_true",
        help="Use the bundled small ActivityNet-style smoke manifest instead of TASK_DATASET_PATH/Hugging Face.",
    )
    parser.add_argument("--split", default=os.getenv("TASK_SPLIT", "validation"))
    parser.add_argument("--hf-cache-dir", default=os.getenv("HF_HOME", ""))
    parser.add_argument(
        "--hf-activitynet-filename",
        default=os.getenv("HF_ACTIVITYNET_FILENAME", ""),
    )
    parser.add_argument(
        "--require-accessible-videos",
        action="store_true",
        default=env_bool("REQUIRE_ACCESSIBLE_VIDEOS", True),
        help="Check source video accessibility before choosing a sample.",
    )
    parser.add_argument(
        "--no-require-accessible-videos",
        action="store_false",
        dest="require_accessible_videos",
        help="Skip the pre-sampling accessibility check.",
    )
    parser.add_argument(
        "--max-sampling-attempts",
        type=int,
        default=env_int("TASK_MAX_SAMPLING_ATTEMPTS", 50),
    )
    parser.add_argument(
        "--validator-hotkey",
        default=os.getenv("VALIDATOR_HOTKEY", "local-hardened-task-verifier"),
    )
    parser.add_argument(
        "--task-secret",
        default=os.getenv("VALIDATOR_TASK_SECRET", ""),
        help="Secret used for task sampling. Defaults to VALIDATOR_TASK_SECRET; an ephemeral local secret is used if unset.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=0,
        help="Block value used in the sampler seed. Defaults to a random local value.",
    )
    parser.add_argument(
        "--query-variants-path",
        default=os.getenv("TASK_QUERY_VARIANTS_PATH", ""),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HARDENED_TASK_VERIFY_CACHE_DIR", DEFAULT_CACHE_DIR),
        help="Local working directory for generated clips and the dry-run upload copy.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Manifest path. Defaults to a fresh per-run manifest under <cache-dir>.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=os.getenv("TASK_ARTIFACT_PREFIX", "task-clips"),
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=env_float("TASK_CLIP_TTL_HOURS", 6.0),
    )
    parser.add_argument(
        "--cleanup-expired",
        action="store_true",
        help="Clean expired manifest entries before generation.",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=env_int("HARDENED_TASK_MAX_GENERATION_ATTEMPTS", 5),
    )
    parser.add_argument(
        "--min-clip-duration-seconds",
        type=float,
        default=env_float("TASK_MIN_CLIP_DURATION_SECONDS", 30.0),
    )
    parser.add_argument(
        "--max-clip-duration-seconds",
        type=float,
        default=env_float("TASK_MAX_CLIP_DURATION_SECONDS", 180.0),
    )
    parser.add_argument(
        "--source-download-timeout-seconds",
        type=int,
        default=env_int("TASK_SOURCE_DOWNLOAD_TIMEOUT_SECONDS", 120),
    )
    parser.add_argument(
        "--encoding-profile-name",
        default=os.getenv("TASK_ENCODING_PROFILE_NAME", "h264-720p-v1"),
    )
    parser.add_argument(
        "--clip-max-width",
        type=int,
        default=env_int("TASK_CLIP_MAX_WIDTH", 1280),
    )
    parser.add_argument(
        "--clip-max-height",
        type=int,
        default=env_int("TASK_CLIP_MAX_HEIGHT", 720),
    )
    parser.add_argument(
        "--clip-video-bitrate",
        default=os.getenv("TASK_CLIP_VIDEO_BITRATE", "1500k"),
    )
    parser.add_argument(
        "--clip-audio-bitrate",
        default=os.getenv("TASK_CLIP_AUDIO_BITRATE", "96k"),
    )
    parser.add_argument(
        "--enable-adversarial-task-transforms",
        action="store_true",
        default=env_bool("ENABLE_ADVERSARIAL_TASK_TRANSFORMS", True),
        help="Enable randomized hardened task transforms.",
    )
    parser.add_argument(
        "--disable-adversarial-task-transforms",
        action="store_false",
        dest="enable_adversarial_task_transforms",
        help="Disable randomized hardened task transforms.",
    )
    parser.add_argument(
        "--enable-task-encoding-profile-variants",
        action="store_true",
        default=env_bool("ENABLE_TASK_ENCODING_PROFILE_VARIANTS", True),
        help="Enable randomized encoding profile variants.",
    )
    parser.add_argument(
        "--disable-task-encoding-profile-variants",
        action="store_false",
        dest="enable_task_encoding_profile_variants",
        help="Disable randomized encoding profile variants.",
    )
    parser.add_argument(
        "--canary-task-rate",
        type=float,
        default=env_float("CANARY_TASK_RATE", 0.0),
        help="Fraction of hardened tasks converted into internal validator canaries.",
    )
    parser.add_argument(
        "--absent-canary-queries",
        default=os.getenv("ABSENT_CANARY_QUERIES", ""),
        help="Pipe-separated absent-query canary prompts.",
    )
    parser.add_argument(
        "--max-active-tasks-per-source-video",
        type=int,
        default=env_int("MAX_ACTIVE_TASKS_PER_SOURCE_VIDEO", 0),
    )
    parser.add_argument(
        "--max-active-tasks-per-source-caption",
        type=int,
        default=env_int("MAX_ACTIVE_TASKS_PER_SOURCE_CAPTION", 0),
    )
    parser.add_argument(
        "--max-active-tasks-per-query-variant",
        type=int,
        default=env_int("MAX_ACTIVE_TASKS_PER_QUERY_VARIANT", 0),
    )
    parser.add_argument(
        "--max-active-tasks-per-transform",
        type=int,
        default=env_int("MAX_ACTIVE_TASKS_PER_TRANSFORM", 0),
    )
    parser.add_argument(
        "--use-vidaio",
        action="store_true",
        default=env_bool("VIDAIO_COMPRESSION_ENABLED", False),
        help="Try Vidaio compression before local ffmpeg, matching validator behavior.",
    )
    parser.add_argument("--vidaio-api-base-url", default=os.getenv("VIDAIO_API_BASE_URL", ""))
    parser.add_argument("--vidaio-api-key", default=os.getenv("VIDAIO_API_KEY", ""))
    parser.add_argument(
        "--vidaio-timeout-seconds",
        type=float,
        default=env_float("VIDAIO_TIMEOUT_SECONDS", 60.0),
    )
    parser.add_argument(
        "--upload-to-hippius",
        action="store_true",
        help="Use real Hippius S3 storage instead of local dry-run storage.",
    )
    parser.add_argument(
        "--skip-hippius-bucket-setup",
        action="store_true",
        help="When uploading to Hippius, skip ensure_bucket_public().",
    )
    parser.add_argument(
        "--hippius-s3-endpoint-url",
        default=os.getenv("HIPPIUS_S3_ENDPOINT_URL", "https://s3.hippius.com"),
    )
    parser.add_argument(
        "--hippius-s3-public-base-url",
        default=os.getenv("HIPPIUS_S3_PUBLIC_BASE_URL", "https://s3.hippius.com"),
    )
    parser.add_argument(
        "--hippius-s3-bucket",
        default=os.getenv("HIPPIUS_S3_BUCKET", "chronoseek"),
    )
    parser.add_argument(
        "--hippius-s3-access-key-id",
        default=os.getenv("HIPPIUS_S3_ACCESS_KEY_ID", ""),
    )
    parser.add_argument(
        "--hippius-s3-secret-access-key",
        default=os.getenv("HIPPIUS_S3_SECRET_ACCESS_KEY", ""),
    )
    parser.add_argument(
        "--hippius-s3-region",
        default=os.getenv("HIPPIUS_S3_REGION", "decentralized"),
    )
    parser.add_argument(
        "--hippius-s3-timeout-seconds",
        type=float,
        default=env_float("HIPPIUS_S3_TIMEOUT_SECONDS", 60.0),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=env_int("VALIDATOR_EVAL_TOP_K", 1),
        help="top_k value used when printing the miner-facing request model.",
    )
    return parser


def resolve_dataset_path(args: argparse.Namespace) -> str:
    if args.use_smoke_dataset:
        return str(
            Path(__file__).resolve().parent.parent
            / "chronoseek"
            / "validator"
            / "data"
            / "smoke_test_tasks.json"
        )
    return args.dataset_path


def build_availability_checker(args: argparse.Namespace) -> VideoAvailabilityChecker | None:
    from chronoseek.validator.video_availability import VideoAvailabilityChecker

    if not args.require_accessible_videos:
        return None
    return VideoAvailabilityChecker(timeout=args.source_download_timeout_seconds)


def build_storage(args: argparse.Namespace, cache_dir: Path):
    if not args.upload_to_hippius:
        return LocalDryRunStorage(str(cache_dir / "dry-run-uploaded"))

    from chronoseek.hippius.s3 import HippiusS3Config, HippiusS3StorageClient

    storage = HippiusS3StorageClient(
        HippiusS3Config(
            endpoint_url=args.hippius_s3_endpoint_url,
            public_base_url=args.hippius_s3_public_base_url,
            bucket=args.hippius_s3_bucket,
            access_key_id=args.hippius_s3_access_key_id,
            secret_access_key=args.hippius_s3_secret_access_key,
            region=args.hippius_s3_region,
        ),
        timeout_seconds=args.hippius_s3_timeout_seconds,
    )
    if not args.skip_hippius_bucket_setup:
        storage.ensure_bucket_public()
    return storage


def build_generator(args: argparse.Namespace):
    from chronoseek.validator.artifact_manifest import TaskArtifactManifest
    from chronoseek.validator.clipper import FfmpegClipper
    from chronoseek.validator.compression import CompositeCompressor, LocalFfmpegCompressor
    from chronoseek.validator.hardened_task_gen import (
        HardenedActivityNetTaskGenerator,
        HardenedTaskGeneratorConfig,
    )
    from chronoseek.validator.query_variants import QueryVariantSelector
    from chronoseek.validator.task_gen import ActivityNetTaskGenerator
    from chronoseek.validator.task_models import EncodingProfile
    from chronoseek.validator.task_sampler import ActivityNetTaskSampler
    from chronoseek.video.vidaio import VidaioCompressor

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest_path).expanduser()
        if args.manifest_path
        else cache_dir / f"artifact_manifest-{int(time.time())}-{secrets.token_hex(4)}.json"
    )
    dataset_path = resolve_dataset_path(args)
    block = args.block or secrets.randbelow(2_000_000_000)
    task_secret = args.task_secret or f"local-verifier-{secrets.token_urlsafe(32)}"

    availability_checker = build_availability_checker(args)
    source_task_gen = ActivityNetTaskGenerator(
        dataset_path=dataset_path or None,
        split=args.split,
        cache_dir=args.hf_cache_dir or None,
        dataset_filename=args.hf_activitynet_filename or None,
        require_accessible_videos=args.require_accessible_videos,
        availability_checker=availability_checker,
        max_sampling_attempts=args.max_sampling_attempts,
    )
    encoding_profile = EncodingProfile(
        name=args.encoding_profile_name,
        max_width=int(args.clip_max_width),
        max_height=int(args.clip_max_height),
        video_bitrate=args.clip_video_bitrate,
        audio_bitrate=args.clip_audio_bitrate,
    )
    sampler = ActivityNetTaskSampler(
        source_task_gen.dataset,
        validator_hotkey=args.validator_hotkey,
        task_secret=task_secret,
        video_cooldown_seconds=0,
        caption_cooldown_seconds=0,
        availability_checker=availability_checker,
        require_accessible_videos=args.require_accessible_videos,
        max_sampling_attempts=args.max_sampling_attempts,
    )
    recording_sampler = RecordingTaskSampler(sampler)
    clipper = FfmpegClipper(
        cache_dir=str(cache_dir / "clips"),
        min_clip_duration_seconds=args.min_clip_duration_seconds,
        max_clip_duration_seconds=args.max_clip_duration_seconds,
        download_timeout_seconds=args.source_download_timeout_seconds,
        encoding_profile=encoding_profile,
    )
    local_compressor = LocalFfmpegCompressor(encoding_profile=encoding_profile)
    vidaio_compressor = None
    if args.use_vidaio:
        vidaio_compressor = VidaioCompressor(
            api_base_url=args.vidaio_api_base_url,
            api_key=args.vidaio_api_key or None,
            timeout_seconds=args.vidaio_timeout_seconds,
            encoding_profile=encoding_profile,
        )
    compressor = CompositeCompressor(
        local_compressor=local_compressor,
        preferred_compressor=vidaio_compressor,
        preferred_enabled=args.use_vidaio,
        preferred_backend_name="Vidaio",
    )
    manifest = TaskArtifactManifest(str(manifest_path))
    storage = build_storage(args, cache_dir)
    generator = HardenedActivityNetTaskGenerator(
        sampler=recording_sampler,
        query_selector=QueryVariantSelector(args.query_variants_path or None),
        clipper=clipper,
        compressor=compressor,
        storage=storage,
        manifest=manifest,
        validator_hotkey=args.validator_hotkey,
        current_block_provider=lambda: block,
        config=HardenedTaskGeneratorConfig(
            artifact_prefix=args.artifact_prefix,
            ttl_hours=args.ttl_hours,
            cleanup_interval_seconds=0,
            max_generation_attempts=args.max_generation_attempts,
            delete_remote_artifacts=False,
            enable_adversarial_transforms=args.enable_adversarial_task_transforms,
            enable_encoding_profile_variants=args.enable_task_encoding_profile_variants,
            canary_task_rate=args.canary_task_rate,
            absent_canary_queries=tuple(
                query.strip()
                for query in str(args.absent_canary_queries or "").split("|")
                if query.strip()
            )
            or HardenedTaskGeneratorConfig.absent_canary_queries,
            max_active_tasks_per_source_video=args.max_active_tasks_per_source_video,
            max_active_tasks_per_source_caption=args.max_active_tasks_per_source_caption,
            max_active_tasks_per_query_variant=args.max_active_tasks_per_query_variant,
            max_active_tasks_per_transform=args.max_active_tasks_per_transform,
        ),
    )
    return {
        "block": block,
        "cache_dir": cache_dir,
        "dataset_path": dataset_path,
        "ephemeral_secret": not bool(args.task_secret),
        "generator": generator,
        "manifest": manifest,
        "recording_sampler": recording_sampler,
        "storage": storage,
    }


def print_json(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = build_parser().parse_args()
    import bittensor as bt

    from chronoseek.protocol_models import VideoSearchRequest
    from chronoseek.validator.task_models import normalize_generated_task

    bt.logging.set_info(True)

    try:
        context = build_generator(args)
        generator = context["generator"]
        manifest = context["manifest"]

        if args.cleanup_expired:
            deleted = generator.cleanup_expired(force=True)
            print(f"Cleaned expired artifacts: {deleted}")

        task = generator.generate_task()
        source_sample = context["recording_sampler"].last_sample
        normalized = normalize_generated_task(task, default_top_k=args.top_k)
        request = VideoSearchRequest(
            request_id=normalized.request_id,
            video={"url": normalized.video_url},
            query=normalized.query,
            top_k=normalized.top_k,
        )
        manifest_entry = manifest.entries.get(task.task_id)
        local_artifact_path = manifest_entry.local_path if manifest_entry else None
        probed_duration = (
            generator.clipper.probe_duration(local_artifact_path)
            if local_artifact_path
            else None
        )
        dry_run_upload_path = None
        storage = context["storage"]
        if isinstance(storage, LocalDryRunStorage):
            dry_run_upload_path = storage.uploads.get(task.artifact_key)

        print_json(
            "Generation Context",
            {
                "block": context["block"],
                "dataset": context["dataset_path"] or "huggingface",
                "ephemeral_secret_used": context["ephemeral_secret"],
                "manifest_path": str(manifest.path),
                "storage": "hippius" if args.upload_to_hippius else "local-dry-run",
                "adversarial_transforms": args.enable_adversarial_task_transforms,
                "encoding_profile_variants": args.enable_task_encoding_profile_variants,
                "canary_task_rate": args.canary_task_rate,
                "exposure_summary": manifest.exposure_summary(now=time.time()),
                "generated_at": iso_timestamp(time.time()),
            },
        )
        print_json(
            "ValidationTask",
            {
                **asdict(task),
                "expires_at_iso": iso_timestamp(task.expires_at),
            },
        )
        if source_sample is not None:
            print_json(
                "Original ActivityNet Task",
                {
                    "source_video_id": source_sample.source_video_id,
                    "source_video_url": source_sample.source_url,
                    "source_caption_id": source_sample.source_caption_id,
                    "description": source_sample.caption,
                    "ground_truths": source_sample.ground_truths,
                    "difficulty": source_sample.difficulty,
                },
            )
        print_json(
            "Miner-Facing Request",
            request.model_dump(mode="json"),
        )
        print_json(
            "Local Artifacts",
            {
                "compressed_artifact_path": local_artifact_path,
                "dry_run_uploaded_copy": dry_run_upload_path,
                "probed_duration_seconds": probed_duration,
            },
        )
        print("\nHardened task generation: ok")
        return 0
    except Exception as exc:
        print(f"\nHardened task generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
