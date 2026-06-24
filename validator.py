"""
ChronoSeek Validator.
Implements synthetic task generation, runtime querying, IoU scoring, and weight updates.
"""

import os
import argparse
import time
import asyncio
import httpx
import bittensor as bt
import threading
import sys
import numpy as np
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chronoseek.validator import task_gen as task_gen_module
from chronoseek.validator import forward as forward_module
from chronoseek.constants import (
    DEFAULT_ABSENT_CANARY_QUERIES,
    DEFAULT_ACCESSIBLE_VIDEO_CACHE_PATH,
    DEFAULT_CANARY_TASK_RATE,
    DEFAULT_CHUTES_BASE_DOMAIN,
    DEFAULT_ENABLE_ADVERSARIAL_TASK_TRANSFORMS,
    DEFAULT_ENABLE_HARDENED_TASKS,
    DEFAULT_ENABLE_LATENCY_MULTIPLIER,
    DEFAULT_ENABLE_TASK_ENCODING_PROFILE_VARIANTS,
    DEFAULT_HF_ACTIVITYNET_FILENAME,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HIPPIUS_S3_BUCKET,
    DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
    DEFAULT_HIPPIUS_S3_PUBLIC_BASE_URL,
    DEFAULT_HIPPIUS_S3_REGION,
    DEFAULT_HIPPIUS_S3_TIMEOUT_SECONDS,
    DEFAULT_HOTKEY_NAME,
    DEFAULT_INACCESSIBLE_VIDEO_CACHE_PATH,
    DEFAULT_LATENCY_GRACE_SECONDS,
    DEFAULT_LATENCY_MIN_MULTIPLIER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_ACTIVE_TASKS_PER_QUERY_VARIANT,
    DEFAULT_MAX_ACTIVE_TASKS_PER_SOURCE_CAPTION,
    DEFAULT_MAX_ACTIVE_TASKS_PER_SOURCE_VIDEO,
    DEFAULT_MAX_ACTIVE_TASKS_PER_TRANSFORM,
    DEFAULT_MINER_EMISSION_BURN_PERCENT,
    DEFAULT_MINER_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_MINER_SUBMISSION_CACHE_TTL_SECONDS,
    DEFAULT_MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS,
    DEFAULT_NETUID,
    DEFAULT_NETWORK,
    DEFAULT_REQUIRE_ACCESSIBLE_VIDEOS,
    DEFAULT_SCORE_CONSISTENCY_WEIGHT,
    DEFAULT_SCORE_EMA_ALPHA,
    DEFAULT_SCORE_RELIABILITY_WEIGHT,
    DEFAULT_SCORE_SUSPICION_WEIGHT,
    DEFAULT_TASK_ARTIFACT_MANIFEST_PATH,
    DEFAULT_TASK_ARTIFACT_PREFIX,
    DEFAULT_TASK_CAPTION_COOLDOWN_HOURS,
    DEFAULT_TASK_CLIP_AUDIO_BITRATE,
    DEFAULT_TASK_CLIP_CACHE_DIR,
    DEFAULT_TASK_CLIP_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_TASK_CLIP_MAX_HEIGHT,
    DEFAULT_TASK_CLIP_MAX_WIDTH,
    DEFAULT_TASK_CLIP_TTL_HOURS,
    DEFAULT_TASK_CLIP_VIDEO_BITRATE,
    DEFAULT_TASK_DATASET_PATH,
    DEFAULT_TASK_DELETE_REMOTE_ARTIFACTS,
    DEFAULT_TASK_ENCODING_PROFILE_NAME,
    DEFAULT_TASK_MAX_CLIP_DURATION_SECONDS,
    DEFAULT_TASK_MAX_PREDICTION_DURATION_SECONDS,
    DEFAULT_TASK_MAX_SAMPLING_ATTEMPTS,
    DEFAULT_TASK_MIN_CLIP_DURATION_SECONDS,
    DEFAULT_TASK_QUERY_VARIANTS_PATH,
    DEFAULT_TASK_SOURCE_DOWNLOAD_TIMEOUT_SECONDS,
    DEFAULT_TASK_SPLIT,
    DEFAULT_TASK_VIDEO_COOLDOWN_HOURS,
    DEFAULT_VALIDATOR_EVAL_TOP_K,
    DEFAULT_VALIDATOR_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_VALIDATOR_TELEMETRY_PATH,
    DEFAULT_VALIDATOR_TASK_SECRET,
    DEFAULT_VIDAIO_API_BASE_URL,
    DEFAULT_VIDAIO_COMPRESSION_ENABLED,
    DEFAULT_VIDAIO_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_AVAILABILITY_CACHE_PATH,
    DEFAULT_VIDEO_AVAILABILITY_CACHE_TTL_HOURS,
    DEFAULT_VIDEO_AVAILABILITY_TIMEOUT,
    DEFAULT_WALLET_NAME,
    DEFAULT_WALLET_PATH,
)
from chronoseek.scoring import LatencyScoringConfig
from chronoseek.hippius.s3 import HippiusS3Config, HippiusS3StorageClient
from chronoseek.validator.artifact_manifest import TaskArtifactManifest
from chronoseek.validator.aggregation import (
    ScoreAggregationConfig,
    update_miner_score,
)
from chronoseek.validator.clipper import FfmpegClipper
from chronoseek.validator.compression import (
    CompositeCompressor,
    LocalFfmpegCompressor,
)
from chronoseek.validator.hardened_task_gen import (
    HardenedActivityNetTaskGenerator,
    HardenedTaskGeneratorConfig,
)
from chronoseek.validator.query_variants import QueryVariantSelector
from chronoseek.validator.state import ValidatorRuntimeState
from chronoseek.validator.task_models import EncodingProfile
from chronoseek.validator.task_sampler import ActivityNetTaskSampler
from chronoseek.video.vidaio import VidaioCompressor
from chronoseek.chain.submissions import (
    MinerSubmissionResolver,
)
from chronoseek.chutes.runtime import (
    build_runtime_endpoints_from_map,
    build_submission_endpoint_map,
    chutes_auth_headers_from_env,
    filter_healthy_runtime_endpoints,
)
from chronoseek.validator.video_availability import VideoAvailabilityChecker


def seed_scores_from_metagraph(metagraph: bt.Metagraph) -> np.ndarray:
    try:
        incentives = getattr(metagraph, "I", None)
        if incentives is None:
            raise ValueError("metagraph incentives are unavailable")

        scores = np.asarray(incentives, dtype=float).reshape(-1)
        if len(scores) != int(metagraph.n):
            raise ValueError(
                f"metagraph incentive length mismatch: expected {metagraph.n}, got {len(scores)}"
            )

        if np.any(np.isnan(scores)) or np.any(scores < 0):
            raise ValueError("metagraph incentives contain invalid values")

        bt.logging.info("Initialized validator scores from metagraph incentives.")
        return np.array(scores, copy=True)
    except Exception as exc:
        bt.logging.warning(
            f"Falling back to zero-initialized validator scores: {exc}"
        )
        return np.zeros(int(metagraph.n), dtype=float)


def heartbeat_monitor(last_heartbeat, stop_event):
    while not stop_event.is_set():
        time.sleep(5)
        if (
            time.time() - last_heartbeat[0]
            > DEFAULT_VALIDATOR_HEARTBEAT_TIMEOUT_SECONDS
        ):
            bt.logging.error(
                "No heartbeat detected in the last "
                f"{DEFAULT_VALIDATOR_HEARTBEAT_TIMEOUT_SECONDS} seconds. "
                "Restarting process."
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)


def get_responsive_uids_snapshot(
    runtime: ValidatorRuntimeState,
) -> tuple[list[int], bool]:
    with runtime.responsive_lock:
        responsive_uids = sorted(runtime.responsive_uids)
        responsive_initialized = runtime.responsive_initialized
    return responsive_uids, responsive_initialized


def get_metagraph_snapshot(runtime: ValidatorRuntimeState):
    with runtime.metagraph_lock:
        return runtime.metagraph


def resize_scores_for_metagraph(
    current_scores: np.ndarray,
    metagraph: bt.Metagraph,
) -> np.ndarray:
    scores = np.array(current_scores, copy=True)
    metagraph_size = int(metagraph.n)
    if len(scores) < metagraph_size:
        new_scores = seed_scores_from_metagraph(metagraph)
        new_scores[: len(scores)] = scores
        return new_scores
    if len(scores) > metagraph_size:
        return scores[:metagraph_size]
    return scores


def replace_runtime_metagraph(
    runtime: ValidatorRuntimeState,
    metagraph: bt.Metagraph,
):
    with runtime.metagraph_lock:
        runtime.metagraph = metagraph


def sync_runtime_metagraph(
    subtensor: bt.Subtensor,
    runtime: ValidatorRuntimeState,
    netuid: int,
) -> bt.Metagraph:
    next_metagraph = bt.Metagraph(
        netuid=netuid,
        network=subtensor.network,
        sync=False,
    )
    next_metagraph.sync(subtensor=subtensor)

    with runtime.score_lock:
        current_scores = np.array(runtime.scores, copy=True)
    resized_scores = resize_scores_for_metagraph(current_scores, next_metagraph)

    replace_runtime_metagraph(runtime, next_metagraph)
    with runtime.score_lock:
        runtime.scores = np.array(resized_scores, copy=True)

    return next_metagraph


def responsive_refresh_due(
    runtime: ValidatorRuntimeState,
    interval_seconds: float,
) -> bool:
    with runtime.responsive_lock:
        if not runtime.responsive_initialized or runtime.responsive_last_refresh_at is None:
            return True
        last_refresh_at = runtime.responsive_last_refresh_at

    return (time.time() - last_refresh_at) >= max(1.0, float(interval_seconds))


async def refresh_responsive_miners_from_submissions(
    runtime: ValidatorRuntimeState,
    subtensor: bt.Subtensor,
    netuid: int,
    submission_resolver: MinerSubmissionResolver,
    chutes_base_domain: str,
    health_timeout_seconds: float,
    provider_headers: dict[str, str] | None = None,
) -> set[int]:
    """
    "Responsive" means:
    1. the registered metagraph hotkey has valid committed runtime metadata
    2. the resolved runtime endpoint currently passes /health
    """
    health_timeout_seconds = max(0.5, float(health_timeout_seconds))
    metagraph = get_metagraph_snapshot(runtime)
    submissions = await submission_resolver.get_submissions(
        subtensor=subtensor,
        netuid=netuid,
        metagraph=metagraph,
    )
    duplicate_hotkeys = set()
    get_duplicate_hotkeys = getattr(
        submission_resolver,
        "get_duplicate_hotkeys",
        None,
    )
    if callable(get_duplicate_hotkeys):
        raw_duplicate_hotkeys = get_duplicate_hotkeys()
        if isinstance(raw_duplicate_hotkeys, (set, list, tuple)):
            duplicate_hotkeys = set(raw_duplicate_hotkeys)
    hotkeys = getattr(metagraph, "hotkeys", [])
    disqualified_uids = {
        int(uid)
        for uid, hotkey in enumerate(hotkeys)
        if str(hotkey) in duplicate_hotkeys
    }
    endpoint_map = build_submission_endpoint_map(
        metagraph=metagraph,
        submissions_by_hotkey=submissions,
        chutes_base_domain=chutes_base_domain,
    )
    healthy_endpoint_map = await filter_healthy_runtime_endpoints(
        endpoint_map=endpoint_map,
        health_timeout_seconds=health_timeout_seconds,
        provider_headers=provider_headers,
    )

    responsive_uids = set(healthy_endpoint_map)
    refreshed_at = time.time()

    with runtime.responsive_lock:
        runtime.responsive_uids = set(responsive_uids)
        runtime.miner_endpoints = dict(healthy_endpoint_map)
        runtime.disqualified_uids = set(disqualified_uids)
        runtime.responsive_initialized = True
        runtime.responsive_last_refresh_at = refreshed_at

    bt.logging.info(
        "Submission metadata refresh completed | "
        f"metadata={len(endpoint_map)}/{len(getattr(metagraph, 'hotkeys', []))} | "
        f"responsive={len(responsive_uids)}/{len(getattr(metagraph, 'hotkeys', []))} | "
        f"disqualified={len(disqualified_uids)} | "
        f"uids={sorted(responsive_uids)}"
    )
    return responsive_uids


def apply_responsive_miner_filter(
    scores: np.ndarray,
    responsive_uids: list[int],
) -> np.ndarray:
    responsive_set = {int(uid) for uid in responsive_uids}
    filtered_scores = np.array(scores, copy=True)
    for uid in range(len(filtered_scores)):
        if uid not in responsive_set:
            filtered_scores[uid] = 0.0
    return filtered_scores


def apply_disqualified_miner_penalty(
    scores: np.ndarray,
    disqualified_uids: list[int] | set[int],
) -> np.ndarray:
    penalized_scores = np.array(scores, copy=True)
    for uid in {int(uid) for uid in disqualified_uids}:
        if 0 <= uid < len(penalized_scores):
            penalized_scores[uid] = 0.0
    return penalized_scores


def build_emission_weights(
    scores: np.ndarray,
    miner_emission_burn_percent: float,
    burn_uid: int = 0,
) -> np.ndarray:
    """
    Reserve a fixed emission share for `burn_uid` and distribute the remaining
    share across all other miners by score.
    """
    weights = np.zeros_like(np.asarray(scores, dtype=float))
    if len(weights) == 0:
        return weights

    burn_percent = normalize_miner_emission_burn_percent(
        miner_emission_burn_percent
    )
    burn_fraction = burn_percent / 100.0
    if 0 <= burn_uid < len(weights):
        weights[burn_uid] = burn_fraction

    distributable_scores = np.array(scores, dtype=float, copy=True)
    if 0 <= burn_uid < len(distributable_scores):
        distributable_scores[burn_uid] = 0.0
    distributable_scores = np.where(distributable_scores > 0, distributable_scores, 0.0)
    distributable_total = float(np.sum(distributable_scores))
    remaining_fraction = max(0.0, 1.0 - burn_fraction)

    if distributable_total > 0 and remaining_fraction > 0:
        weights += (distributable_scores / distributable_total) * remaining_fraction
    elif remaining_fraction > 0 and 0 <= burn_uid < len(weights):
        weights[burn_uid] = 1.0

    return weights


def normalize_miner_emission_burn_percent(value: float) -> float:
    return max(0.0, min(float(value), 100.0))


def get_config_float(config: bt.Config, name: str, default: float) -> float:
    value = getattr(config, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_config_str(config: bt.Config, name: str, default: str = "") -> str:
    value = getattr(config, name, default)
    return value if isinstance(value, str) else default


def get_config_bool(config: bt.Config, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def build_validator_task_generator(
    *,
    config: bt.Config,
    availability_checker: VideoAvailabilityChecker,
    validator_hotkey: str,
    current_block_provider,
):
    source_task_gen = task_gen_module.ActivityNetTaskGenerator(
        dataset_path=config.task_dataset_path or None,
        split=config.task_split,
        cache_dir=config.hf_cache_dir or None,
        dataset_filename=config.hf_activitynet_filename or None,
        require_accessible_videos=config.require_accessible_videos,
        availability_checker=availability_checker,
        max_sampling_attempts=config.task_max_sampling_attempts,
    )

    if not get_config_bool(config, "enable_hardened_tasks", False):
        bt.logging.info("Validator task generation configured | mode=legacy-activitynet")
        return source_task_gen

    task_secret = str(config.validator_task_secret or "").strip()
    if not task_secret:
        raise ValueError(
            "Hardened task generation requires VALIDATOR_TASK_SECRET "
            "or --validator-task-secret."
        )
    hippius_access_key_id = os.getenv("HIPPIUS_S3_ACCESS_KEY_ID", "").strip()
    hippius_secret_access_key = os.getenv("HIPPIUS_S3_SECRET_ACCESS_KEY", "").strip()
    vidaio_api_key = os.getenv("VIDAIO_API_KEY", "").strip() or None

    task_cache_dir = Path(config.task_clip_cache_dir).expanduser()
    task_cache_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest_path = str(config.task_artifact_manifest_path or "").strip()
    if raw_manifest_path:
        manifest_path = Path(raw_manifest_path).expanduser()
    else:
        manifest_path = task_cache_dir / "artifact_manifest.json"

    encoding_profile = EncodingProfile(
        name=config.task_encoding_profile_name,
        max_width=int(config.task_clip_max_width),
        max_height=int(config.task_clip_max_height),
        video_bitrate=config.task_clip_video_bitrate,
        audio_bitrate=config.task_clip_audio_bitrate,
    )
    sampler = ActivityNetTaskSampler(
        source_task_gen.dataset,
        validator_hotkey=validator_hotkey,
        task_secret=task_secret,
        video_cooldown_seconds=float(config.task_video_cooldown_hours) * 3600.0,
        caption_cooldown_seconds=float(config.task_caption_cooldown_hours) * 3600.0,
        availability_checker=availability_checker,
        require_accessible_videos=config.require_accessible_videos,
        max_sampling_attempts=config.task_max_sampling_attempts,
    )
    clipper = FfmpegClipper(
        cache_dir=str(task_cache_dir),
        min_clip_duration_seconds=config.task_min_clip_duration_seconds,
        max_clip_duration_seconds=config.task_max_clip_duration_seconds,
        download_timeout_seconds=config.task_source_download_timeout_seconds,
        encoding_profile=encoding_profile,
    )
    local_compressor = LocalFfmpegCompressor(encoding_profile=encoding_profile)
    vidaio_compressor = None
    if get_config_bool(config, "vidaio_compression_enabled", False):
        vidaio_compressor = VidaioCompressor(
            api_base_url=config.vidaio_api_base_url,
            api_key=vidaio_api_key,
            timeout_seconds=config.vidaio_timeout_seconds,
            encoding_profile=encoding_profile,
        )
    compressor = CompositeCompressor(
        local_compressor=local_compressor,
        preferred_compressor=vidaio_compressor,
        preferred_enabled=get_config_bool(config, "vidaio_compression_enabled", False),
        preferred_backend_name="Vidaio",
    )
    storage = HippiusS3StorageClient(
        HippiusS3Config(
            endpoint_url=config.hippius_s3_endpoint_url,
            public_base_url=config.hippius_s3_public_base_url,
            bucket=config.hippius_s3_bucket,
            access_key_id=hippius_access_key_id,
            secret_access_key=hippius_secret_access_key,
            region=config.hippius_s3_region,
        ),
        timeout_seconds=config.hippius_s3_timeout_seconds,
    )
    storage.ensure_bucket_public()
    bt.logging.info(
        f"Hippius task bucket ready and public-readable | bucket={config.hippius_s3_bucket}"
    )
    manifest = TaskArtifactManifest(str(manifest_path))
    hardened_gen = HardenedActivityNetTaskGenerator(
        sampler=sampler,
        query_selector=QueryVariantSelector(config.task_query_variants_path or None),
        clipper=clipper,
        compressor=compressor,
        storage=storage,
        manifest=manifest,
        validator_hotkey=validator_hotkey,
        current_block_provider=current_block_provider,
        config=HardenedTaskGeneratorConfig(
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
            absent_canary_queries=tuple(
                query.strip()
                for query in str(config.absent_canary_queries or "").split("|")
                if query.strip()
            )
            or HardenedTaskGeneratorConfig.absent_canary_queries,
            max_active_tasks_per_source_video=int(
                config.max_active_tasks_per_source_video
            ),
            max_active_tasks_per_source_caption=int(
                config.max_active_tasks_per_source_caption
            ),
            max_active_tasks_per_query_variant=int(
                config.max_active_tasks_per_query_variant
            ),
            max_active_tasks_per_transform=int(config.max_active_tasks_per_transform),
        ),
    )
    hardened_gen.cleanup_expired(force=True)
    bt.logging.info("Validator task generation configured | mode=hardened-activitynet")
    return hardened_gen


async def run_validator_loop(
    subtensor: bt.Subtensor,
    runtime: ValidatorRuntimeState,
    netuid: int,
    stop_event: threading.Event,
    last_heartbeat: List[float],
    config: bt.Config,
):
    """
    Async validator evaluation loop.
    """
    cache_root = Path.home() / ".cache" / "chronoseek"
    accessible_video_cache_path = config.accessible_video_cache_path
    if not accessible_video_cache_path:
        if config.video_availability_cache_path:
            base_path = Path(config.video_availability_cache_path).expanduser()
            accessible_video_cache_path = str(
                base_path.with_name(
                    f"{base_path.stem}_accessible{base_path.suffix or '.json'}"
                )
            )
        else:
            accessible_video_cache_path = str(cache_root / "accessible_videos.json")

    inaccessible_video_cache_path = config.inaccessible_video_cache_path
    if not inaccessible_video_cache_path:
        if config.video_availability_cache_path:
            base_path = Path(config.video_availability_cache_path).expanduser()
            inaccessible_video_cache_path = str(
                base_path.with_name(
                    f"{base_path.stem}_inaccessible{base_path.suffix or '.json'}"
                )
            )
        else:
            inaccessible_video_cache_path = str(
                cache_root / "inaccessible_videos.json"
            )

    availability_checker = VideoAvailabilityChecker(
        accessible_cache_path=accessible_video_cache_path,
        inaccessible_cache_path=inaccessible_video_cache_path,
        cache_ttl_seconds=int(config.video_availability_cache_ttl_hours * 3600),
        timeout=config.video_availability_timeout,
    )
    task_gen = build_validator_task_generator(
        config=config,
        availability_checker=availability_checker,
        validator_hotkey=runtime.wallet.hotkey.ss58_address,
        current_block_provider=lambda: subtensor.get_current_block(),
    )
    chutes_base_domain = get_config_str(config, "chutes_base_domain", "chutes.ai")
    submission_resolver = MinerSubmissionResolver(
        cache_ttl_seconds=get_config_float(
            config,
            "miner_submission_cache_ttl_seconds",
            300.0,
        ),
    )
    provider_headers = chutes_auth_headers_from_env()
    if provider_headers:
        bt.logging.info(
            "Chutes/provider authorization header is configured for v2 evaluation requests."
        )
    bt.logging.info("Synthetic evaluation routing configured | mode=chain")
    async_client = httpx.AsyncClient(timeout=30.0)
    tempo = subtensor.get_subnet_hyperparameters(netuid).tempo
    bt.logging.info(f"Subnet tempo: {tempo} blocks")
    last_weight_block = 0

    try:
        while not stop_event.is_set():
            current_block = subtensor.get_current_block()
            last_heartbeat[0] = time.time()
            if current_block % 100 == 0 and runtime.last_metagraph_sync_block != current_block:
                sync_runtime_metagraph(subtensor, runtime, netuid)
                runtime.last_metagraph_sync_block = current_block

            metagraph = get_metagraph_snapshot(runtime)
            with runtime.score_lock:
                scores = resize_scores_for_metagraph(runtime.scores, metagraph)

            if responsive_refresh_due(
                runtime,
                float(config.miner_submission_refresh_interval_seconds),
            ):
                await refresh_responsive_miners_from_submissions(
                    runtime=runtime,
                    subtensor=subtensor,
                    netuid=netuid,
                    submission_resolver=submission_resolver,
                    chutes_base_domain=chutes_base_domain,
                    health_timeout_seconds=get_config_float(
                        config,
                        "miner_submission_health_timeout_seconds",
                        10.0,
                    ),
                    provider_headers=provider_headers,
                )

            metagraph = get_metagraph_snapshot(runtime)
            with runtime.score_lock:
                scores = resize_scores_for_metagraph(runtime.scores, metagraph)
            with runtime.responsive_lock:
                endpoint_map = dict(runtime.miner_endpoints)
                candidate_uids = sorted(
                    uid for uid in runtime.responsive_uids if uid in endpoint_map
                )
                disqualified_uids = set(runtime.disqualified_uids)

            miner_endpoints = build_runtime_endpoints_from_map(
                metagraph=metagraph,
                endpoint_map=endpoint_map,
                candidate_uids=candidate_uids,
            )
            if disqualified_uids:
                scores = apply_disqualified_miner_penalty(scores, disqualified_uids)
                with runtime.score_lock:
                    runtime.scores = np.array(scores, copy=True)
            scores = apply_responsive_miner_filter(scores, candidate_uids)

            if not candidate_uids:
                bt.logging.warning(
                    "Skipping validation step because no responsive miners have valid committed metadata and healthy runtimes."
                )
                await asyncio.sleep(12)
                continue
            # --- 1. Run Validation Step ---
            step_scores = await forward_module.run_step(
                task_gen,
                metagraph,
                runtime.wallet,
                async_client,
                miner_timeout_seconds=float(config.miner_request_timeout_seconds),
                miner_endpoints=miner_endpoints,
                provider_headers=provider_headers,
                validator_eval_top_k=int(config.validator_eval_top_k),
                max_prediction_duration_seconds=float(
                    config.task_max_prediction_duration_seconds
                ),
                latency_scoring_config=LatencyScoringConfig(
                    enabled=get_config_bool(config, "enable_latency_multiplier", False),
                    grace_seconds=get_config_float(
                        config,
                        "latency_grace_seconds",
                        30.0,
                    ),
                    timeout_seconds=float(config.miner_request_timeout_seconds),
                    min_multiplier=get_config_float(
                        config,
                        "latency_min_multiplier",
                        0.85,
                    ),
                ),
                telemetry_recorder=runtime.telemetry.record,
            )

            # Update aggregate scores. Component weights default to zero, which
            # preserves the historical quality-only EMA behavior.
            aggregation_config = ScoreAggregationConfig(
                alpha=get_config_float(config, "score_ema_alpha", 0.1),
                reliability_weight=get_config_float(
                    config,
                    "score_reliability_weight",
                    0.0,
                ),
                consistency_weight=get_config_float(
                    config,
                    "score_consistency_weight",
                    0.0,
                ),
                suspicion_weight=get_config_float(
                    config,
                    "score_suspicion_weight",
                    0.0,
                ),
            )
            aggregate_components = {}
            for uid, score in step_scores:
                if uid < len(scores):
                    components = update_miner_score(
                        previous_score=float(scores[uid]),
                        instant_score=float(score),
                        telemetry_summary=runtime.telemetry.summaries.get(int(uid)),
                        config=aggregation_config,
                    )
                    scores[uid] = components.final_score
                    aggregate_components[int(uid)] = components
            with runtime.score_lock:
                runtime.scores = np.array(scores, copy=True)

            if step_scores:
                ranked_step_scores = sorted(
                    step_scores, key=lambda item: item[1], reverse=True
                )
                step_summary = ", ".join(
                    f"UID {uid}: {score:.4f}" for uid, score in ranked_step_scores[:10]
                )
                bt.logging.info(f"Step scores: {step_summary}")

                ranked_moving_scores = sorted(
                    (
                        (uid, float(scores[uid]))
                        for uid, _ in step_scores
                        if uid < len(scores)
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                moving_summary = ", ".join(
                    f"UID {uid}: {score:.4f}"
                    for uid, score in ranked_moving_scores[:10]
                )
                bt.logging.info(f"Moving scores: {moving_summary}")
                if aggregate_components:
                    component_summary = ", ".join(
                        (
                            f"UID {uid}: quality={components.quality_score:.4f} "
                            f"reliability={components.reliability_score:.2f} "
                            f"consistency={components.consistency_score:.2f} "
                            f"suspicion={components.suspicion_score:.2f}"
                        )
                        for uid, components in sorted(aggregate_components.items())[:10]
                    )
                    bt.logging.info(f"Score components: {component_summary}")

                telemetry_path = get_config_str(
                    config,
                    "validator_telemetry_path",
                    "",
                ).strip()
                if telemetry_path:
                    runtime.telemetry.save_json(telemetry_path)

                telemetry_snapshot = runtime.telemetry.snapshot()
                summary_items = telemetry_snapshot.get("summaries", {})
                flagged = [
                    (uid, payload.get("suspicion_flags", []))
                    for uid, payload in summary_items.items()
                    if payload.get("suspicion_flags")
                ]
                if flagged:
                    bt.logging.warning(f"Telemetry suspicion flags: {flagged[:10]}")

            # --- 2. Set Weights ---
            blocks_since_last = current_block - last_weight_block
            if blocks_since_last >= tempo:
                bt.logging.info(
                    f"Block {current_block}: Setting weights (tempo={tempo})"
                )

                burn_percent = normalize_miner_emission_burn_percent(
                    config.miner_emission_burn_percent
                )
                weights = build_emission_weights(
                    scores,
                    miner_emission_burn_percent=burn_percent,
                    burn_uid=0,
                )
                bt.logging.info(
                    "Emission weights prepared | "
                    f"burn_uid=0 | burn_percent={burn_percent:.2f} | "
                    f"distributed_percent={100.0 - burn_percent:.2f}"
                )

                # Convert to lists
                uids_list = list(range(len(weights)))
                weights_list = weights.tolist()

                # Set weights on chain
                try:
                    success = subtensor.set_weights(
                        wallet=runtime.wallet,
                        netuid=netuid,
                        uids=uids_list,
                        weights=weights_list,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                    )
                    if success:
                        bt.logging.success("Successfully set weights")
                        last_weight_block = current_block
                except Exception as e:
                    bt.logging.error(f"Failed to set weights: {e}")

            # Sleep a bit
            await asyncio.sleep(12)

    finally:
        if async_client is not None:
            await async_client.aclose()


def get_config():
    """
    Parse arguments and return configuration.
    Priority: CLI > explicit operator environment > code defaults.

    Most validator tuning values intentionally default from constants instead
    of environment variables so operators only need to provide identity,
    credentials, and secrets in `.env`.
    """
    parser = argparse.ArgumentParser(description="ChronoSeek Validator")

    # Add bittensor arguments first
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Add custom arguments
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", str(DEFAULT_NETUID))),
        help="Subnet NetUID",
    )
    parser.add_argument(
        "--task-dataset-path",
        type=str,
        default=DEFAULT_TASK_DATASET_PATH,
        help="Optional local task dataset path. If omitted, ActivityNet is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default=DEFAULT_TASK_SPLIT,
        help="Task split to use when loading validator data.",
    )
    parser.add_argument(
        "--require-accessible-videos",
        action="store_true",
        default=DEFAULT_REQUIRE_ACCESSIBLE_VIDEOS,
        help="Skip validator tasks whose source videos are not currently accessible.",
    )
    parser.add_argument(
        "--no-require-accessible-videos",
        action="store_false",
        dest="require_accessible_videos",
        help="Skip validator-side source video accessibility checks.",
    )
    parser.add_argument(
        "--task-max-sampling-attempts",
        type=int,
        default=DEFAULT_TASK_MAX_SAMPLING_ATTEMPTS,
        help="Maximum random samples to try before giving up on finding an accessible validator task.",
    )
    parser.add_argument(
        "--enable-hardened-tasks",
        action="store_true",
        default=DEFAULT_ENABLE_HARDENED_TASKS,
        help="Generate private clipped ActivityNet tasks and upload them to Hippius before querying miners.",
    )
    parser.add_argument(
        "--disable-hardened-tasks",
        action="store_false",
        dest="enable_hardened_tasks",
        help="Use legacy ActivityNet URL tasks instead of private clipped task artifacts.",
    )
    parser.add_argument(
        "--validator-task-secret",
        type=str,
        default=os.getenv("VALIDATOR_TASK_SECRET", DEFAULT_VALIDATOR_TASK_SECRET),
        help="Private hardened-task sampling secret. Defaults to VALIDATOR_TASK_SECRET.",
    )
    parser.add_argument(
        "--hardened-task-max-generation-attempts",
        type=int,
        default=DEFAULT_HARDENED_TASK_MAX_GENERATION_ATTEMPTS,
        help="Maximum hardened task generation attempts before skipping a validator step.",
    )
    parser.add_argument(
        "--task-query-variants-path",
        type=str,
        default=DEFAULT_TASK_QUERY_VARIANTS_PATH,
        help="Private query variant manifest path. Exact ActivityNet captions are avoided when variants exist.",
    )
    parser.add_argument(
        "--task-clip-cache-dir",
        type=str,
        default=DEFAULT_TASK_CLIP_CACHE_DIR,
        help="Local cache directory for generated validator task clips.",
    )
    parser.add_argument(
        "--task-artifact-manifest-path",
        type=str,
        default=DEFAULT_TASK_ARTIFACT_MANIFEST_PATH,
        help="Local JSON manifest of uploaded validator task artifacts.",
    )
    parser.add_argument(
        "--task-artifact-prefix",
        type=str,
        default=DEFAULT_TASK_ARTIFACT_PREFIX,
        help="Object key prefix for uploaded validator task clips.",
    )
    parser.add_argument(
        "--task-clip-ttl-hours",
        type=float,
        default=DEFAULT_TASK_CLIP_TTL_HOURS,
        help="TTL in hours for local and Hippius validator task artifacts.",
    )
    parser.add_argument(
        "--task-clip-cleanup-interval-seconds",
        type=float,
        default=DEFAULT_TASK_CLIP_CLEANUP_INTERVAL_SECONDS,
        help="Minimum interval between expired validator task artifact cleanup runs.",
    )
    parser.add_argument(
        "--task-delete-remote-artifacts",
        action="store_true",
        default=DEFAULT_TASK_DELETE_REMOTE_ARTIFACTS,
        help="Delete expired uploaded validator task clips from Hippius during cleanup.",
    )
    parser.add_argument(
        "--task-video-cooldown-hours",
        type=float,
        default=DEFAULT_TASK_VIDEO_COOLDOWN_HOURS,
        help="Cooldown before the same ActivityNet source video may be sampled again.",
    )
    parser.add_argument(
        "--task-caption-cooldown-hours",
        type=float,
        default=DEFAULT_TASK_CAPTION_COOLDOWN_HOURS,
        help="Cooldown before the same ActivityNet source caption may be sampled again.",
    )
    parser.add_argument(
        "--task-min-clip-duration-seconds",
        type=float,
        default=DEFAULT_TASK_MIN_CLIP_DURATION_SECONDS,
        help="Minimum generated task clip duration.",
    )
    parser.add_argument(
        "--task-max-clip-duration-seconds",
        type=float,
        default=DEFAULT_TASK_MAX_CLIP_DURATION_SECONDS,
        help="Maximum generated task clip duration.",
    )
    parser.add_argument(
        "--task-source-download-timeout-seconds",
        type=int,
        default=DEFAULT_TASK_SOURCE_DOWNLOAD_TIMEOUT_SECONDS,
        help="Timeout in seconds when downloading ActivityNet source videos for clipping.",
    )
    parser.add_argument(
        "--task-encoding-profile-name",
        type=str,
        default=DEFAULT_TASK_ENCODING_PROFILE_NAME,
        help="Name recorded for the validator task encoding profile.",
    )
    parser.add_argument(
        "--task-clip-max-width",
        type=int,
        default=DEFAULT_TASK_CLIP_MAX_WIDTH,
        help="Maximum generated task clip width.",
    )
    parser.add_argument(
        "--task-clip-max-height",
        type=int,
        default=DEFAULT_TASK_CLIP_MAX_HEIGHT,
        help="Maximum generated task clip height.",
    )
    parser.add_argument(
        "--task-clip-video-bitrate",
        type=str,
        default=DEFAULT_TASK_CLIP_VIDEO_BITRATE,
        help="Generated task clip video bitrate.",
    )
    parser.add_argument(
        "--task-clip-audio-bitrate",
        type=str,
        default=DEFAULT_TASK_CLIP_AUDIO_BITRATE,
        help="Generated task clip audio bitrate.",
    )
    parser.add_argument(
        "--enable-adversarial-task-transforms",
        action="store_true",
        default=DEFAULT_ENABLE_ADVERSARIAL_TASK_TRANSFORMS,
        help="Enable randomized hardened task transforms such as context and encoding variation.",
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
        default=DEFAULT_ENABLE_TASK_ENCODING_PROFILE_VARIANTS,
        help="Enable randomized encoding profile variants for hardened task clips.",
    )
    parser.add_argument(
        "--disable-task-encoding-profile-variants",
        action="store_false",
        dest="enable_task_encoding_profile_variants",
        help="Disable randomized encoding profile variants for hardened task clips.",
    )
    parser.add_argument(
        "--canary-task-rate",
        type=float,
        default=DEFAULT_CANARY_TASK_RATE,
        help="Fraction of hardened tasks converted into internal validator canaries.",
    )
    parser.add_argument(
        "--absent-canary-queries",
        type=str,
        default=DEFAULT_ABSENT_CANARY_QUERIES,
        help="Pipe-separated absent-query canary prompts. Defaults to built-in unlikely prompts.",
    )
    parser.add_argument(
        "--max-active-tasks-per-source-video",
        type=int,
        default=DEFAULT_MAX_ACTIVE_TASKS_PER_SOURCE_VIDEO,
        help="Maximum active hardened artifacts per source video. 0 disables the limit.",
    )
    parser.add_argument(
        "--max-active-tasks-per-source-caption",
        type=int,
        default=DEFAULT_MAX_ACTIVE_TASKS_PER_SOURCE_CAPTION,
        help="Maximum active hardened artifacts per source caption. 0 disables the limit.",
    )
    parser.add_argument(
        "--max-active-tasks-per-query-variant",
        type=int,
        default=DEFAULT_MAX_ACTIVE_TASKS_PER_QUERY_VARIANT,
        help="Maximum active hardened artifacts per query variant. 0 disables the limit.",
    )
    parser.add_argument(
        "--max-active-tasks-per-transform",
        type=int,
        default=DEFAULT_MAX_ACTIVE_TASKS_PER_TRANSFORM,
        help="Maximum active hardened artifacts per encoding transform. 0 disables the limit.",
    )
    parser.add_argument(
        "--validator-eval-top-k",
        type=int,
        default=DEFAULT_VALIDATOR_EVAL_TOP_K,
        help="Number of miner results requested and scored for synthetic evaluation.",
    )
    parser.add_argument(
        "--score-ema-alpha",
        type=float,
        default=DEFAULT_SCORE_EMA_ALPHA,
        help="EMA alpha used for instant miner quality scores.",
    )
    parser.add_argument(
        "--score-reliability-weight",
        type=float,
        default=DEFAULT_SCORE_RELIABILITY_WEIGHT,
        help="Optional reliability multiplier weight applied after quality scoring.",
    )
    parser.add_argument(
        "--score-consistency-weight",
        type=float,
        default=DEFAULT_SCORE_CONSISTENCY_WEIGHT,
        help="Optional consistency multiplier weight applied after quality scoring.",
    )
    parser.add_argument(
        "--score-suspicion-weight",
        type=float,
        default=DEFAULT_SCORE_SUSPICION_WEIGHT,
        help="Optional suspicion multiplier weight applied after quality scoring.",
    )
    parser.add_argument(
        "--task-max-prediction-duration-seconds",
        type=float,
        default=DEFAULT_TASK_MAX_PREDICTION_DURATION_SECONDS,
        help="Maximum scored miner interval duration unless the ground-truth interval is longer.",
    )
    parser.add_argument(
        "--enable-latency-multiplier",
        action="store_true",
        default=DEFAULT_ENABLE_LATENCY_MULTIPLIER,
        help="Apply a gentle latency multiplier after interval quality scoring.",
    )
    parser.add_argument(
        "--latency-grace-seconds",
        type=float,
        default=DEFAULT_LATENCY_GRACE_SECONDS,
        help="Latency before the optional multiplier begins decaying.",
    )
    parser.add_argument(
        "--latency-min-multiplier",
        type=float,
        default=DEFAULT_LATENCY_MIN_MULTIPLIER,
        help="Lowest multiplier applied near the miner request timeout.",
    )
    parser.add_argument(
        "--hippius-s3-endpoint-url",
        type=str,
        default=DEFAULT_HIPPIUS_S3_ENDPOINT_URL,
        help="Hippius S3-compatible endpoint URL used for validator task uploads.",
    )
    parser.add_argument(
        "--hippius-s3-public-base-url",
        type=str,
        default=DEFAULT_HIPPIUS_S3_PUBLIC_BASE_URL,
        help="Public base URL for uploaded Hippius validator task clips.",
    )
    parser.add_argument(
        "--hippius-s3-bucket",
        type=str,
        default=DEFAULT_HIPPIUS_S3_BUCKET,
        help="Hippius S3 bucket for validator task clips.",
    )
    parser.add_argument(
        "--hippius-s3-region",
        type=str,
        default=DEFAULT_HIPPIUS_S3_REGION,
        help="S3 signing region for Hippius uploads.",
    )
    parser.add_argument(
        "--hippius-s3-timeout-seconds",
        type=float,
        default=DEFAULT_HIPPIUS_S3_TIMEOUT_SECONDS,
        help="Hippius S3 upload/delete timeout in seconds.",
    )
    parser.add_argument(
        "--vidaio-compression-enabled",
        action="store_true",
        default=DEFAULT_VIDAIO_COMPRESSION_ENABLED,
        help="Try Vidaio compression before falling back to local ffmpeg.",
    )
    parser.add_argument(
        "--vidaio-api-base-url",
        type=str,
        default=os.getenv("VIDAIO_API_BASE_URL", DEFAULT_VIDAIO_API_BASE_URL),
        help="Optional Vidaio compression API base URL.",
    )
    parser.add_argument(
        "--vidaio-timeout-seconds",
        type=float,
        default=DEFAULT_VIDAIO_TIMEOUT_SECONDS,
        help="Vidaio compression timeout in seconds.",
    )
    parser.add_argument(
        "--video-availability-cache-path",
        type=str,
        default=DEFAULT_VIDEO_AVAILABILITY_CACHE_PATH,
        help="Legacy base path used to derive accessible/inaccessible validator video cache files when explicit cache paths are not configured.",
    )
    parser.add_argument(
        "--accessible-video-cache-path",
        type=str,
        default=DEFAULT_ACCESSIBLE_VIDEO_CACHE_PATH,
        help="Path to a JSON cache of validator videos confirmed to be publicly accessible.",
    )
    parser.add_argument(
        "--inaccessible-video-cache-path",
        type=str,
        default=DEFAULT_INACCESSIBLE_VIDEO_CACHE_PATH,
        help="Path to a JSON cache of validator videos confirmed to be inaccessible.",
    )
    parser.add_argument(
        "--video-availability-cache-ttl-hours",
        type=float,
        default=DEFAULT_VIDEO_AVAILABILITY_CACHE_TTL_HOURS,
        help="TTL in hours for cached validator video availability checks.",
    )
    parser.add_argument(
        "--video-availability-timeout",
        type=int,
        default=DEFAULT_VIDEO_AVAILABILITY_TIMEOUT,
        help="Timeout in seconds for validator-side video availability checks.",
    )
    parser.add_argument(
        "--miner-request-timeout-seconds",
        type=float,
        default=DEFAULT_MINER_REQUEST_TIMEOUT_SECONDS,
        help="Per-miner timeout in seconds for validator runtime search requests.",
    )
    parser.add_argument(
        "--miner-submission-cache-ttl-seconds",
        type=float,
        default=DEFAULT_MINER_SUBMISSION_CACHE_TTL_SECONDS,
        help="TTL for cached v2 miner submissions loaded from chain.",
    )
    parser.add_argument(
        "--miner-submission-refresh-interval-seconds",
        type=float,
        default=DEFAULT_MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS,
        help="Interval in seconds between validator refreshes of miner submission metadata.",
    )
    parser.add_argument(
        "--miner-submission-health-timeout-seconds",
        type=float,
        default=DEFAULT_MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS,
        help="Per-runtime timeout for /health checks during responsive miner refresh.",
    )
    parser.add_argument(
        "--validator-telemetry-path",
        type=str,
        default=DEFAULT_VALIDATOR_TELEMETRY_PATH,
        help="Optional JSON path for validator miner behavior telemetry snapshots.",
    )
    parser.add_argument(
        "--chutes-base-domain",
        type=str,
        default=DEFAULT_CHUTES_BASE_DOMAIN,
        help="Base Chutes domain used to resolve chute_slug submissions into HTTPS endpoints.",
    )
    parser.add_argument(
        "--miner-emission-burn-percent",
        type=float,
        default=DEFAULT_MINER_EMISSION_BURN_PERCENT,
        help="Percent of miner emissions to burn by assigning that weight share to UID 0. The remaining share is distributed to UID 1+ by score.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=DEFAULT_HF_CACHE_DIR,
        help="Optional Hugging Face cache directory for validator dataset downloads.",
    )
    parser.add_argument(
        "--hf-activitynet-filename",
        type=str,
        default=DEFAULT_HF_ACTIVITYNET_FILENAME,
        help="Optional filename override inside the ActivityNet Hugging Face snapshot.",
    )

    # Set defaults from environment variables for bittensor arguments
    # Note: We must use the internal argument names (dest) which usually replace dots with underscores
    # However, bittensor seems to use dots in dest names, so we use a dict to set defaults
    defaults = {
        "wallet.name": os.getenv("WALLET_NAME", DEFAULT_WALLET_NAME),
        "wallet.hotkey": os.getenv("HOTKEY_NAME", DEFAULT_HOTKEY_NAME),
        "wallet.path": os.getenv("WALLET_PATH", DEFAULT_WALLET_PATH),
        "subtensor.network": os.getenv("NETWORK", DEFAULT_NETWORK),
        "logging.level": os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
    }

    parser.set_defaults(**defaults)

    return bt.Config(parser)


def main():
    """Run the ChronoSeek validator."""
    # 1. Configuration
    config = get_config()

    # Setup logging
    bt.logging(config=config, logging_dir=config.logging.logging_dir)
    bt.logging.on()  # Ensure console logging is on

    # Force debug if requested, otherwise default to INFO
    if config.logging.level == "DEBUG":
        bt.logging.set_debug(True)
    elif config.logging.level == "TRACE":
        bt.logging.set_trace(True)
    else:
        # Default to INFO if not specified
        bt.logging.set_info(True)

    bt.logging.info(
        f"Starting ChronoSeek Validator on network={config.subtensor.network}, netuid={config.netuid}"
    )
    bt.logging.info(f"Full config: {config}")

    # Heartbeat setup
    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_monitor, args=(last_heartbeat, stop_event), daemon=True
    )
    heartbeat_thread.start()

    try:
        # 2. Setup
        wallet = bt.Wallet(config=config)
        bt.logging.info(f"Wallet: {wallet}")

        try:
            if wallet.hotkey:
                bt.logging.info(
                    f"Starting Validator with hotkey: {wallet.hotkey.ss58_address}"
                )
        except Exception as e:
            bt.logging.error(f"Error checking wallet: {e}")
            stop_event.set()
            return

        subtensor = bt.Subtensor(config=config)
        metagraph = bt.Metagraph(
            netuid=config.netuid, network=subtensor.network, sync=False
        )
        metagraph.sync(subtensor=subtensor)

        # 3. Check Registration
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(
                f"Validator hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}"
            )
            stop_event.set()
            return

        bt.logging.info(
            f"Validator registered with UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}"
        )

        chutes_base_domain = get_config_str(config, "chutes_base_domain", "chutes.ai")
        provider_headers = chutes_auth_headers_from_env()
        runtime = ValidatorRuntimeState(
            wallet=wallet,
            metagraph=metagraph,
            scores=seed_scores_from_metagraph(metagraph),
            score_lock=threading.Lock(),
            provider_headers=provider_headers,
        )

        sync_runtime_metagraph(subtensor, runtime, config.netuid)
        submission_resolver = MinerSubmissionResolver(
            cache_ttl_seconds=get_config_float(
                config,
                "miner_submission_cache_ttl_seconds",
                300.0,
            )
        )
        initial_responsive_uids = asyncio.run(
            refresh_responsive_miners_from_submissions(
                runtime=runtime,
                subtensor=subtensor,
                netuid=config.netuid,
                submission_resolver=submission_resolver,
                chutes_base_domain=chutes_base_domain,
                health_timeout_seconds=get_config_float(
                    config,
                    "miner_submission_health_timeout_seconds",
                    10.0,
                ),
                provider_headers=provider_headers,
            )
        )
        runtime.scores = apply_responsive_miner_filter(
            apply_disqualified_miner_penalty(
                runtime.scores,
                runtime.disqualified_uids,
            ),
            sorted(initial_responsive_uids),
        )
        bt.logging.info(
            f"Initialized responsive miner snapshot with {len(initial_responsive_uids)} miners."
        )

        asyncio.run(
            run_validator_loop(
                subtensor,
                runtime,
                config.netuid,
                stop_event,
                last_heartbeat,
                config,
            )
        )

    except KeyboardInterrupt:
        bt.logging.info("Validator stopped by user")
    except Exception as e:
        bt.logging.error(f"Fatal error: {e}")
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
