"""
ChronoSeek Validator.
Implements synthetic task generation, runtime querying, IoU scoring, and weight updates.
"""

import os
import argparse
import time
import asyncio
import httpx
import threading
import sys
import numpy as np
from typing import Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


from chronoseek.validator import forward as forward_module
from chronoseek.bittensor_sdk import (
    add_bittensor_arguments,
    create_subtensor,
    create_wallet,
    current_block,
    fetch_metagraph,
    parse_config,
    set_weights as set_chain_weights,
    subnet_tempo,
)
from chronoseek.constants import (
    DEFAULT_ABSENT_CANARY_QUERIES,
    DEFAULT_ACCESSIBLE_VIDEO_CACHE_PATH,
    DEFAULT_CANARY_TASK_RATE,
    DEFAULT_CHUTES_BASE_DOMAIN,
    DEFAULT_ENABLE_ADVERSARIAL_TASK_TRANSFORMS,
    DEFAULT_ENABLE_HARDENED_TASKS,
    DEFAULT_ENABLE_LATENCY_MULTIPLIER,
    DEFAULT_ENABLE_TASK_ENCODING_PROFILE_VARIANTS,
    DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
    DEFAULT_EVALUATION_ROUND_BLOCKS,
    DEFAULT_HF_ACTIVITYNET_FILENAME,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_HARDENED_TASK_MAX_GENERATION_ATTEMPTS,
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
    DEFAULT_ENABLE_PROOF_OF_ACCESS,
    DEFAULT_PROOF_OF_ACCESS_CACHE_TTL_HOURS,
    DEFAULT_PROOF_OF_ACCESS_TIMEOUT_SECONDS,
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
    DEFAULT_VIDAIO_POLL_INTERVAL_SECONDS,
    DEFAULT_VIDAIO_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_AVAILABILITY_CACHE_PATH,
    DEFAULT_VIDEO_AVAILABILITY_CACHE_TTL_HOURS,
    DEFAULT_VIDEO_AVAILABILITY_TIMEOUT,
    DEFAULT_WALLET_NAME,
    DEFAULT_WALLET_PATH,
)
from chronoseek.scoring import LatencyScoringConfig
from chronoseek.logging import configure_logging, logger
from chronoseek.validator.aggregation import (
    ScoreAggregationConfig,
    update_miner_score,
)
from chronoseek.validator.config_values import (
    get_config_bool,
    get_config_float,
    get_config_str,
)
from chronoseek.validator.state import ValidatorRuntimeState
from chronoseek.validator.task_generator_factory import build_validator_task_generator
from chronoseek.chain.submissions import (
    MinerSubmissionResolver,
)
from chronoseek.chutes.runtime import (
    build_runtime_endpoints_from_map,
    build_submission_endpoint_map,
    chutes_auth_headers_from_env,
    filter_healthy_runtime_endpoints,
    filter_proof_of_access_passed,
)
from chronoseek.validator.video_availability import VideoAvailabilityChecker
from chronoseek.video.downloader import VideoDownloader
from chronoseek.video.proof_of_access import compute_proof_of_access_hash


def seed_scores_from_metagraph(metagraph: Any) -> np.ndarray:
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

        logger.info("Initialized validator scores from metagraph incentives.")
        return np.array(scores, copy=True)
    except Exception as exc:
        logger.warning(
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
            logger.error(
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
    metagraph: Any,
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
    metagraph: Any,
):
    with runtime.metagraph_lock:
        runtime.metagraph = metagraph


def sync_runtime_metagraph(
    subtensor: Any,
    runtime: ValidatorRuntimeState,
    netuid: int,
) -> Any:
    next_metagraph = fetch_metagraph(subtensor, netuid)

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


def pick_proof_of_access_reference_video(task_gen: Any) -> str | None:
    """
    Reuse a video the validator already vets for hardened task generation as
    the proof-of-access reference, instead of maintaining a separate canary
    list.
    """
    dataset = getattr(task_gen, "dataset", None) or []
    for entry in dataset:
        video_url = entry.get("video_url") if isinstance(entry, dict) else None
        if video_url:
            return str(video_url)
    return None


async def refresh_proof_of_access(
    runtime: ValidatorRuntimeState,
    *,
    healthy_endpoint_map: dict[int, str],
    task_gen: Any,
    cache_ttl_seconds: float,
    timeout_seconds: float,
    provider_headers: dict[str, str] | None = None,
) -> set[int]:
    """
    Returns the subset of `healthy_endpoint_map`'s uids that currently have a
    valid (non-expired) passing proof-of-access cache entry - checking any
    uid whose cache entry is missing or expired first. New miners (nothing
    cached yet) are always checked immediately, not given a free pass.
    """
    now = time.time()
    cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))

    with runtime.proof_of_access_lock:
        cache = dict(runtime.proof_of_access_cache)

    def is_expired(uid: int) -> bool:
        entry = cache.get(uid)
        if not entry:
            return True
        return (now - float(entry.get("checked_at", 0))) > cache_ttl_seconds

    uids_needing_check = [uid for uid in healthy_endpoint_map if is_expired(uid)]

    if uids_needing_check:
        reference_video_url = pick_proof_of_access_reference_video(task_gen)
        expected_hash = None
        if reference_video_url:
            downloaded_video = VideoDownloader.download_video(
                reference_video_url,
                raise_on_failure=False,
            )
            if downloaded_video is not None:
                try:
                    expected_hash = compute_proof_of_access_hash(downloaded_video.path)
                finally:
                    VideoDownloader.cleanup(downloaded_video)

        if expected_hash:
            check_endpoint_map = {
                uid: healthy_endpoint_map[uid] for uid in uids_needing_check
            }
            passed_uids = await filter_proof_of_access_passed(
                endpoint_map=check_endpoint_map,
                wallet=runtime.wallet,
                video_url=reference_video_url,
                expected_hash=expected_hash,
                timeout_seconds=timeout_seconds,
                provider_headers=provider_headers,
            )
            for uid in uids_needing_check:
                cache[int(uid)] = {"passed": uid in passed_uids, "checked_at": now}
        else:
            logger.warning(
                "Proof-of-access reference video could not be prepared this cycle; "
                "skipping newly-due checks (existing cached passes are unaffected)."
            )

    with runtime.proof_of_access_lock:
        runtime.proof_of_access_cache.update(cache)
        final_cache = dict(runtime.proof_of_access_cache)

    return {
        int(uid)
        for uid in healthy_endpoint_map
        if final_cache.get(int(uid), {}).get("passed")
        and (now - float(final_cache[int(uid)].get("checked_at", 0))) <= cache_ttl_seconds
    }


async def refresh_responsive_miners_from_submissions(
    runtime: ValidatorRuntimeState,
    subtensor: Any,
    netuid: int,
    submission_resolver: MinerSubmissionResolver,
    chutes_base_domain: str,
    health_timeout_seconds: float,
    provider_headers: dict[str, str] | None = None,
    task_gen: Any = None,
    enable_proof_of_access: bool = True,
    proof_of_access_cache_ttl_seconds: float = 21600.0,
    proof_of_access_timeout_seconds: float = 120.0,
) -> set[int]:
    """
    "Responsive" means:
    1. the registered metagraph hotkey has valid committed runtime metadata
    2. the resolved runtime endpoint currently passes /health
    3. the resolved runtime endpoint has a valid (non-expired) passing
       proof-of-access check, when `task_gen` is supplied and proof-of-access
       is enabled - skipped entirely otherwise (e.g. the initial pre-loop
       snapshot, which has no task generator yet)
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

    if enable_proof_of_access and task_gen is not None and healthy_endpoint_map:
        passed_uids = await refresh_proof_of_access(
            runtime,
            healthy_endpoint_map=healthy_endpoint_map,
            task_gen=task_gen,
            cache_ttl_seconds=proof_of_access_cache_ttl_seconds,
            timeout_seconds=proof_of_access_timeout_seconds,
            provider_headers=provider_headers,
        )
        healthy_endpoint_map = {
            uid: endpoint
            for uid, endpoint in healthy_endpoint_map.items()
            if uid in passed_uids
        }

    responsive_uids = set(healthy_endpoint_map)
    refreshed_at = time.time()

    with runtime.responsive_lock:
        runtime.responsive_uids = set(responsive_uids)
        runtime.miner_endpoints = dict(healthy_endpoint_map)
        runtime.disqualified_uids = set(disqualified_uids)
        runtime.responsive_initialized = True
        runtime.responsive_last_refresh_at = refreshed_at

    logger.info(
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


TOP3_SLOT_SHARES = (0.75, 0.20, 0.05)


def top3_weight_shares(scores: np.ndarray) -> np.ndarray:
    """
    Rank-based 75/20/5 weight shares for the top 3 positive scores.

    A miner with score <= 0 never receives weight, even if it would
    otherwise rank in the top 3. When several miners tie on score for a
    paid slot, the weight of every slot their tie spans is pooled and
    split evenly across all of them - including ties larger than 3 (e.g.
    a 5-way tie for 1st splits the full 75+20+5 pool five ways).
    """
    shares = np.zeros(len(scores), dtype=float)

    ranked = sorted((i for i in range(len(scores)) if scores[i] > 0), key=lambda i: scores[i], reverse=True)

    position = 0
    i = 0
    while i < len(ranked) and position < len(TOP3_SLOT_SHARES):
        group_score = scores[ranked[i]]
        group = [ranked[i]]
        j = i + 1
        while j < len(ranked) and scores[ranked[j]] == group_score:
            group.append(ranked[j])
            j += 1

        overlap_end = min(position + len(group), len(TOP3_SLOT_SHARES))
        pooled = sum(TOP3_SLOT_SHARES[position:overlap_end])
        share_each = pooled / len(group)
        for idx in group:
            shares[idx] = share_each

        position += len(group)
        i = j

    return shares


def build_emission_weights(
    scores: np.ndarray,
    miner_emission_burn_percent: float,
    burn_uid: int = 0,
) -> np.ndarray:
    """
    Reserve a fixed emission share for `burn_uid`, then pay the remaining
    share to the top 3 miners by score (75/20/5) - everyone else gets 0.
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
    remaining_fraction = max(0.0, 1.0 - burn_fraction)

    top3_shares = top3_weight_shares(distributable_scores)
    if remaining_fraction > 0 and np.any(top3_shares > 0):
        weights += top3_shares * remaining_fraction
    elif remaining_fraction > 0 and 0 <= burn_uid < len(weights):
        weights[burn_uid] = 1.0

    return weights


def normalize_miner_emission_burn_percent(value: float) -> float:
    return max(0.0, min(float(value), 100.0))


async def run_validator_loop(
    subtensor: Any,
    runtime: ValidatorRuntimeState,
    netuid: int,
    stop_event: threading.Event,
    last_heartbeat: List[float],
    config: Any,
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
        current_block_provider=lambda: current_block(subtensor),
    )
    chutes_base_domain = get_config_str(config, "chutes_base_domain", "chutes.ai")
    submission_resolver = MinerSubmissionResolver(
        cache_ttl_seconds=get_config_float(
            config,
            "miner_submission_cache_ttl_seconds",
            300.0,
        ),
        enforce_one_hotkey_one_submission=get_config_bool(
            config,
            "enforce_one_hotkey_one_submission",
            DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
        ),
    )
    provider_headers = chutes_auth_headers_from_env()
    if provider_headers:
        logger.info(
            "Chutes/provider authorization header is configured for v2 evaluation requests."
        )
    logger.info("Synthetic evaluation routing configured | mode=chain")
    async_client = httpx.AsyncClient(timeout=30.0)
    tempo = subnet_tempo(subtensor, netuid)
    logger.info(f"Subnet tempo: {tempo} blocks")
    last_weight_block = 0
    last_round_start_block = 0
    evaluation_round_blocks = int(config.evaluation_round_blocks)

    try:
        while not stop_event.is_set():
            current_block_number = current_block(subtensor)
            last_heartbeat[0] = time.time()
            if (
                current_block_number % 100 == 0
                and runtime.last_metagraph_sync_block != current_block_number
            ):
                sync_runtime_metagraph(subtensor, runtime, netuid)
                runtime.last_metagraph_sync_block = current_block_number

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
                    task_gen=task_gen,
                    enable_proof_of_access=get_config_bool(
                        config,
                        "enable_proof_of_access",
                        DEFAULT_ENABLE_PROOF_OF_ACCESS,
                    ),
                    proof_of_access_cache_ttl_seconds=get_config_float(
                        config,
                        "proof_of_access_cache_ttl_hours",
                        DEFAULT_PROOF_OF_ACCESS_CACHE_TTL_HOURS,
                    )
                    * 3600.0,
                    proof_of_access_timeout_seconds=get_config_float(
                        config,
                        "proof_of_access_timeout_seconds",
                        DEFAULT_PROOF_OF_ACCESS_TIMEOUT_SECONDS,
                    ),
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
                logger.warning(
                    "Skipping validation step because no responsive miners have valid committed metadata and healthy runtimes."
                )
                await asyncio.sleep(12)
                continue

            # Round pacing is block-based: a round starts every
            # evaluation_round_blocks since the previous round's start, or
            # immediately if a slow round already blew past that boundary.
            if current_block_number - last_round_start_block < evaluation_round_blocks:
                await asyncio.sleep(12)
                continue
            last_round_start_block = current_block_number

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
                logger.info(f"Step scores: {step_summary}")

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
                logger.info(f"Moving scores: {moving_summary}")
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
                    logger.info(f"Score components: {component_summary}")

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
                    logger.warning(f"Telemetry suspicion flags: {flagged[:10]}")

            # --- 2. Set Weights ---
            blocks_since_last = current_block_number - last_weight_block
            if blocks_since_last >= tempo:
                logger.info(
                    f"Block {current_block_number}: Setting weights (tempo={tempo})"
                )

                burn_percent = normalize_miner_emission_burn_percent(
                    config.miner_emission_burn_percent
                )
                weights = build_emission_weights(
                    scores,
                    miner_emission_burn_percent=burn_percent,
                    burn_uid=0,
                )
                logger.info(
                    "Emission weights prepared | "
                    f"burn_uid=0 | burn_percent={burn_percent:.2f} | "
                    f"distributed_percent={100.0 - burn_percent:.2f}"
                )

                # Convert to lists
                uids_list = list(range(len(weights)))
                weights_list = weights.tolist()

                # Set weights on chain
                try:
                    success = set_chain_weights(
                        subtensor=subtensor,
                        wallet=runtime.wallet,
                        netuid=netuid,
                        uids=uids_list,
                        weights=weights_list,
                    )
                    if success:
                        logger.success("Successfully set weights")
                        last_weight_block = current_block_number
                except Exception as e:
                    logger.error(f"Failed to set weights: {e}")

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

    add_bittensor_arguments(parser)

    # Add custom arguments
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", str(DEFAULT_NETUID))),
        help="Subnet NetUID",
    )
    parser.add_argument(
        "--enforce-one-hotkey-one-submission",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENFORCE_ONE_HOTKEY_ONE_SUBMISSION",
            DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
        ),
        help="Disqualify hotkeys with multiple revealed submissions. Enabled by default.",
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
        "--evaluation-round-blocks",
        type=int,
        default=int(
            os.getenv("EVALUATION_ROUND_BLOCKS", str(DEFAULT_EVALUATION_ROUND_BLOCKS))
        ),
        help="Minimum blocks between the start of consecutive evaluation rounds.",
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
        default=os.getenv("HIPPIUS_S3_BUCKET", DEFAULT_HIPPIUS_S3_BUCKET),
        help="Hippius S3 bucket for validator task clips. Defaults to HIPPIUS_S3_BUCKET.",
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
        default=env_bool(
            "VIDAIO_COMPRESSION_ENABLED", DEFAULT_VIDAIO_COMPRESSION_ENABLED
        ),
        help="Try Vidaio compression before falling back to local ffmpeg. Enabled by default.",
    )
    parser.add_argument(
        "--disable-vidaio-compression",
        action="store_false",
        dest="vidaio_compression_enabled",
        help="Skip Vidaio compression and use local ffmpeg compression directly.",
    )
    parser.add_argument(
        "--vidaio-api-base-url",
        type=str,
        default=DEFAULT_VIDAIO_API_BASE_URL,
        help="Vidaio compression API base URL.",
    )
    parser.add_argument(
        "--vidaio-timeout-seconds",
        type=float,
        default=env_float("VIDAIO_TIMEOUT_SECONDS", DEFAULT_VIDAIO_TIMEOUT_SECONDS),
        help="Vidaio compression timeout in seconds.",
    )
    parser.add_argument(
        "--vidaio-poll-interval-seconds",
        type=float,
        default=env_float(
            "VIDAIO_POLL_INTERVAL_SECONDS", DEFAULT_VIDAIO_POLL_INTERVAL_SECONDS
        ),
        help="Seconds between Vidaio workflow status reads.",
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
        "--disable-proof-of-access",
        dest="enable_proof_of_access",
        action="store_false",
        default=env_bool("ENABLE_PROOF_OF_ACCESS", DEFAULT_ENABLE_PROOF_OF_ACCESS),
        help="Disable the periodic proof-of-access check (requires a miner to actually download/decode a real video).",
    )
    parser.add_argument(
        "--proof-of-access-cache-ttl-hours",
        type=float,
        default=env_float(
            "PROOF_OF_ACCESS_CACHE_TTL_HOURS", DEFAULT_PROOF_OF_ACCESS_CACHE_TTL_HOURS
        ),
        help="Hours a passing proof-of-access check remains valid before a miner is re-checked.",
    )
    parser.add_argument(
        "--proof-of-access-timeout-seconds",
        type=float,
        default=env_float(
            "PROOF_OF_ACCESS_TIMEOUT_SECONDS", DEFAULT_PROOF_OF_ACCESS_TIMEOUT_SECONDS
        ),
        help="Per-runtime timeout for /proof-of-access checks (the miner must download and decode a real video).",
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

    return parse_config(parser)


def main():
    """Run the ChronoSeek validator."""
    # 1. Configuration
    config = get_config()

    configure_logging(
        config.logging.level,
        logging_dir=config.logging.logging_dir,
        filename="validator.log",
    )

    logger.info(
        f"Starting ChronoSeek Validator on network={config.subtensor.network}, netuid={config.netuid}"
    )
    logger.info(f"Full config: {config}")

    # Heartbeat setup
    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_monitor, args=(last_heartbeat, stop_event), daemon=True
    )
    heartbeat_thread.start()

    try:
        # 2. Setup
        wallet = create_wallet(config)
        logger.info(f"Wallet: {wallet}")

        try:
            if wallet.hotkey:
                logger.info(
                    f"Starting Validator with hotkey: {wallet.hotkey.ss58_address}"
                )
        except Exception as e:
            logger.error(f"Error checking wallet: {e}")
            stop_event.set()
            return

        subtensor = create_subtensor(config)
        metagraph = fetch_metagraph(subtensor, config.netuid)

        # 3. Check Registration
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            logger.error(
                f"Validator hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}"
            )
            stop_event.set()
            return

        logger.info(
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
            ),
            enforce_one_hotkey_one_submission=get_config_bool(
                config,
                "enforce_one_hotkey_one_submission",
                DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
            ),
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
        logger.info(
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
        logger.info("Validator stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


if __name__ == "__main__":
    main()
