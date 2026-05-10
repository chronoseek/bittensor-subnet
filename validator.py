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
from chronoseek.validator.state import ValidatorRuntimeState
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

HEARTBEAT_TIMEOUT = 600  # seconds


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
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            bt.logging.error(
                "No heartbeat detected in the last 600 seconds. Restarting process."
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
        runtime.responsive_initialized = True
        runtime.responsive_last_refresh_at = refreshed_at

    bt.logging.info(
        "Submission metadata refresh completed | "
        f"metadata={len(endpoint_map)}/{len(getattr(metagraph, 'hotkeys', []))} | "
        f"responsive={len(responsive_uids)}/{len(getattr(metagraph, 'hotkeys', []))} | "
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
    task_gen = task_gen_module.ActivityNetTaskGenerator(
        dataset_path=config.task_dataset_path or None,
        split=config.task_split,
        cache_dir=config.hf_cache_dir or None,
        dataset_filename=config.hf_activitynet_filename or None,
        require_accessible_videos=config.require_accessible_videos,
        availability_checker=availability_checker,
        max_sampling_attempts=config.task_max_sampling_attempts,
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

            miner_endpoints = build_runtime_endpoints_from_map(
                metagraph=metagraph,
                endpoint_map=endpoint_map,
                candidate_uids=candidate_uids,
            )
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
            )

            # Update moving average scores
            alpha = 0.1
            for uid, score in step_scores:
                if uid < len(scores):
                    scores[uid] = alpha * score + (1 - alpha) * scores[uid]
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
    Priority: CLI > Environment Variables > Defaults
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
        default=int(os.getenv("NETUID", "1")),
        help="Subnet NetUID",
    )
    parser.add_argument(
        "--task-dataset-path",
        type=str,
        default=os.getenv("TASK_DATASET_PATH", ""),
        help="Optional local task dataset path. If omitted, ActivityNet is loaded from Hugging Face.",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default=os.getenv("TASK_SPLIT", "validation"),
        help="Task split to use when loading validator data.",
    )
    parser.add_argument(
        "--require-accessible-videos",
        action="store_true",
        default=os.getenv("REQUIRE_ACCESSIBLE_VIDEOS", "1")
        not in {"0", "false", "False"},
        help="Skip validator tasks whose source videos are not currently accessible.",
    )
    parser.add_argument(
        "--task-max-sampling-attempts",
        type=int,
        default=int(os.getenv("TASK_MAX_SAMPLING_ATTEMPTS", "50")),
        help="Maximum random samples to try before giving up on finding an accessible validator task.",
    )
    parser.add_argument(
        "--video-availability-cache-path",
        type=str,
        default=os.getenv("VIDEO_AVAILABILITY_CACHE_PATH", ""),
        help="Legacy base path used to derive accessible/inaccessible validator video cache files when explicit cache paths are not configured.",
    )
    parser.add_argument(
        "--accessible-video-cache-path",
        type=str,
        default=os.getenv("ACCESSIBLE_VIDEO_CACHE_PATH", ""),
        help="Path to a JSON cache of validator videos confirmed to be publicly accessible.",
    )
    parser.add_argument(
        "--inaccessible-video-cache-path",
        type=str,
        default=os.getenv("INACCESSIBLE_VIDEO_CACHE_PATH", ""),
        help="Path to a JSON cache of validator videos confirmed to be inaccessible.",
    )
    parser.add_argument(
        "--video-availability-cache-ttl-hours",
        type=float,
        default=float(os.getenv("VIDEO_AVAILABILITY_CACHE_TTL_HOURS", "24")),
        help="TTL in hours for cached validator video availability checks.",
    )
    parser.add_argument(
        "--video-availability-timeout",
        type=int,
        default=int(os.getenv("VIDEO_AVAILABILITY_TIMEOUT", "20")),
        help="Timeout in seconds for validator-side video availability checks.",
    )
    parser.add_argument(
        "--miner-request-timeout-seconds",
        type=float,
        default=float(os.getenv("MINER_REQUEST_TIMEOUT_SECONDS", "150")),
        help="Per-miner timeout in seconds for validator runtime search requests.",
    )
    parser.add_argument(
        "--miner-submission-cache-ttl-seconds",
        type=float,
        default=float(os.getenv("MINER_SUBMISSION_CACHE_TTL_SECONDS", "300")),
        help="TTL for cached v2 miner submissions loaded from chain.",
    )
    parser.add_argument(
        "--miner-submission-refresh-interval-seconds",
        type=float,
        default=float(os.getenv("MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS", "60")),
        help="Interval in seconds between validator refreshes of miner submission metadata.",
    )
    parser.add_argument(
        "--miner-submission-health-timeout-seconds",
        type=float,
        default=float(os.getenv("MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS", "10")),
        help="Per-runtime timeout for /health checks during responsive miner refresh.",
    )
    parser.add_argument(
        "--chutes-base-domain",
        type=str,
        default=os.getenv("CHUTES_BASE_DOMAIN", "chutes.ai"),
        help="Base Chutes domain used to resolve chute_slug submissions into HTTPS endpoints.",
    )
    parser.add_argument(
        "--miner-emission-burn-percent",
        type=float,
        default=float(os.getenv("MINER_EMISSION_BURN_PERCENT", "0")),
        help="Percent of miner emissions to burn by assigning that weight share to UID 0. The remaining share is distributed to UID 1+ by score.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=os.getenv("HF_HOME", ""),
        help="Optional Hugging Face cache directory for validator dataset downloads.",
    )
    parser.add_argument(
        "--hf-activitynet-filename",
        type=str,
        default=os.getenv("HF_ACTIVITYNET_FILENAME", ""),
        help="Optional filename override inside the ActivityNet Hugging Face snapshot.",
    )

    # Set defaults from environment variables for bittensor arguments
    # Note: We must use the internal argument names (dest) which usually replace dots with underscores
    # However, bittensor seems to use dots in dest names, so we use a dict to set defaults
    defaults = {
        "wallet.name": os.getenv("WALLET_NAME", "default"),
        "wallet.hotkey": os.getenv("HOTKEY_NAME", "default"),
        "wallet.path": os.getenv("WALLET_PATH", "~/.bittensor/wallets/"),
        "subtensor.network": os.getenv("NETWORK", "finney"),
        "logging.level": os.getenv("LOG_LEVEL", "INFO"),
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
            runtime.scores,
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
