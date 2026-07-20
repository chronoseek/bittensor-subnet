import os
from pathlib import Path
from typing import Any, Callable

from chronoseek.constants import DEFAULT_VIDAIO_COMPRESSION_ENABLED
from chronoseek.hippius.s3 import HippiusS3Config, HippiusS3StorageClient
from chronoseek.logging import logger
from chronoseek.validator import task_gen as task_gen_module
from chronoseek.validator.artifact_manifest import TaskArtifactManifest
from chronoseek.validator.clipper import FfmpegClipper
from chronoseek.validator.compression import CompositeCompressor, LocalFfmpegCompressor
from chronoseek.validator.config_values import get_config_bool
from chronoseek.validator.hardened_task_gen import (
    HardenedActivityNetTaskGenerator,
    HardenedTaskGeneratorConfig,
)
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.task_models import EncodingProfile
from chronoseek.validator.task_sampler import ActivityNetTaskSampler
from chronoseek.validator.video_availability import VideoAvailabilityChecker
from chronoseek.video.vidaio import StorageBackedVidaioCompressor, VidaioCompressor


def _manifest_path(config: Any, cache_dir: Path) -> Path:
    configured_path = str(config.task_artifact_manifest_path or "").strip()
    if configured_path:
        return Path(configured_path).expanduser()
    return cache_dir / "artifact_manifest.json"


def _encoding_profile(config: Any) -> EncodingProfile:
    return EncodingProfile(
        name=config.task_encoding_profile_name,
        max_width=int(config.task_clip_max_width),
        max_height=int(config.task_clip_max_height),
        video_bitrate=config.task_clip_video_bitrate,
        audio_bitrate=config.task_clip_audio_bitrate,
    )


def _hardened_config(config: Any) -> HardenedTaskGeneratorConfig:
    absent_queries = tuple(
        query.strip()
        for query in str(config.absent_canary_queries or "").split("|")
        if query.strip()
    )
    return HardenedTaskGeneratorConfig(
        artifact_prefix=config.task_artifact_prefix,
        ttl_hours=config.task_clip_ttl_hours,
        cleanup_interval_seconds=config.task_clip_cleanup_interval_seconds,
        max_generation_attempts=config.hardened_task_max_generation_attempts,
        delete_remote_artifacts=get_config_bool(
            config,
            "task_delete_remote_artifacts",
            False,
        ),
        enable_adversarial_transforms=get_config_bool(
            config,
            "enable_adversarial_task_transforms",
            True,
        ),
        enable_encoding_profile_variants=get_config_bool(
            config,
            "enable_task_encoding_profile_variants",
            True,
        ),
        canary_task_rate=float(config.canary_task_rate),
        absent_canary_queries=(
            absent_queries or HardenedTaskGeneratorConfig.absent_canary_queries
        ),
        max_active_tasks_per_source_video=int(config.max_active_tasks_per_source_video),
        max_active_tasks_per_source_caption=int(config.max_active_tasks_per_source_caption),
        max_active_tasks_per_query_variant=int(config.max_active_tasks_per_query_variant),
        max_active_tasks_per_transform=int(config.max_active_tasks_per_transform),
    )


def _storage(config: Any) -> HippiusS3StorageClient:
    storage = HippiusS3StorageClient(
        HippiusS3Config(
            endpoint_url=config.hippius_s3_endpoint_url,
            public_base_url=config.hippius_s3_public_base_url,
            bucket=config.hippius_s3_bucket,
            access_key_id=os.getenv("HIPPIUS_S3_ACCESS_KEY_ID", "").strip(),
            secret_access_key=os.getenv("HIPPIUS_S3_SECRET_ACCESS_KEY", "").strip(),
            region=config.hippius_s3_region,
        ),
        timeout_seconds=config.hippius_s3_timeout_seconds,
    )
    storage.ensure_bucket_public()
    logger.info(
        f"Hippius task bucket ready and public-readable | bucket={config.hippius_s3_bucket}"
    )
    return storage


def _compressor(
    *,
    config: Any,
    storage: HippiusS3StorageClient,
    encoding_profile: EncodingProfile,
) -> CompositeCompressor:
    vidaio_enabled = get_config_bool(
        config,
        "vidaio_compression_enabled",
        DEFAULT_VIDAIO_COMPRESSION_ENABLED,
    )
    vidaio_compressor = None
    if vidaio_enabled:
        vidaio_compressor = StorageBackedVidaioCompressor(
            vidaio_compressor=VidaioCompressor(
                api_base_url=config.vidaio_api_base_url,
                api_key=os.getenv("VIDAIO_API_KEY", "").strip() or None,
                timeout_seconds=config.vidaio_timeout_seconds,
                poll_interval_seconds=config.vidaio_poll_interval_seconds,
                encoding_profile=encoding_profile,
            ),
            storage=storage,
            artifact_prefix=f"{config.task_artifact_prefix.strip('/')}/vidaio-inputs",
            delete_remote_artifacts=True,
        )
    return CompositeCompressor(
        local_compressor=LocalFfmpegCompressor(encoding_profile=encoding_profile),
        preferred_compressor=vidaio_compressor,
        preferred_enabled=vidaio_enabled,
        preferred_backend_name="Vidaio",
    )


def build_validator_task_generator(
    *,
    config: Any,
    availability_checker: VideoAvailabilityChecker,
    validator_hotkey: str,
    current_block_provider: Callable[[], int],
):
    source_task_generator = task_gen_module.ActivityNetTaskGenerator(
        dataset_path=config.task_dataset_path or None,
        split=config.task_split,
        cache_dir=config.hf_cache_dir or None,
        dataset_filename=config.hf_activitynet_filename or None,
        require_accessible_videos=config.require_accessible_videos,
        availability_checker=availability_checker,
        max_sampling_attempts=config.task_max_sampling_attempts,
    )
    if not get_config_bool(config, "enable_hardened_tasks", False):
        logger.info("Validator task generation configured | mode=legacy-activitynet")
        return source_task_generator

    task_secret = str(config.validator_task_secret or "").strip()
    if not task_secret:
        raise ValueError(
            "Hardened task generation requires VALIDATOR_TASK_SECRET "
            "or --validator-task-secret."
        )

    cache_dir = Path(config.task_clip_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoding_profile = _encoding_profile(config)
    storage = _storage(config)
    generator = HardenedActivityNetTaskGenerator(
        sampler=ActivityNetTaskSampler(
            source_task_generator.dataset,
            validator_hotkey=validator_hotkey,
            task_secret=task_secret,
            video_cooldown_seconds=float(config.task_video_cooldown_hours) * 3600.0,
            caption_cooldown_seconds=float(config.task_caption_cooldown_hours) * 3600.0,
            availability_checker=availability_checker,
            require_accessible_videos=config.require_accessible_videos,
            max_sampling_attempts=config.task_max_sampling_attempts,
        ),
        query_selector=QueryVariantSelector(config.task_query_variants_path or None),
        clipper=FfmpegClipper(
            cache_dir=str(cache_dir),
            min_clip_duration_seconds=config.task_min_clip_duration_seconds,
            max_clip_duration_seconds=config.task_max_clip_duration_seconds,
            download_timeout_seconds=config.task_source_download_timeout_seconds,
            encoding_profile=encoding_profile,
        ),
        compressor=_compressor(
            config=config,
            storage=storage,
            encoding_profile=encoding_profile,
        ),
        storage=storage,
        manifest=TaskArtifactManifest(str(_manifest_path(config, cache_dir))),
        validator_hotkey=validator_hotkey,
        current_block_provider=current_block_provider,
        config=_hardened_config(config),
    )
    generator.cleanup_expired(force=True)
    logger.info("Validator task generation configured | mode=hardened-activitynet")
    return generator
