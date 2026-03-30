"""
ChronoSeek Validator.
Implements the synthetic task generation, miner querying (HTTP+Epistula), and IoU scoring loop.
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

# Use ActivityNetTaskGenerator for MVP
from chronoseek.validator import task_gen as task_gen_module
from chronoseek.validator import forward as forward_module
from chronoseek.validator.gateway import ValidatorGatewayRuntime, create_validator_gateway
from chronoseek.validator.video_availability import VideoAvailabilityChecker

HEARTBEAT_TIMEOUT = 600  # seconds


def heartbeat_monitor(last_heartbeat, stop_event):
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            bt.logging.error(
                "No heartbeat detected in the last 600 seconds. Restarting process."
            )
            os.execv(sys.executable, [sys.executable] + sys.argv)


async def run_validator_loop(
    subtensor: bt.Subtensor,
    runtime: ValidatorGatewayRuntime,
    netuid: int,
    stop_event: threading.Event,
    last_heartbeat: List[float],
    config: bt.Config,
):
    """
    Async validator loop.
    """
    # Initialize components
    cache_root = Path.home() / ".cache" / "chronoseek"
    accessible_video_cache_path = config.accessible_video_cache_path
    if not accessible_video_cache_path:
        if config.video_availability_cache_path:
            base_path = Path(config.video_availability_cache_path).expanduser()
            accessible_video_cache_path = str(
                base_path.with_name(f"{base_path.stem}_accessible{base_path.suffix or '.json'}")
            )
        else:
            accessible_video_cache_path = str(cache_root / "accessible_videos.json")

    inaccessible_video_cache_path = config.inaccessible_video_cache_path
    if not inaccessible_video_cache_path:
        if config.video_availability_cache_path:
            base_path = Path(config.video_availability_cache_path).expanduser()
            inaccessible_video_cache_path = str(
                base_path.with_name(f"{base_path.stem}_inaccessible{base_path.suffix or '.json'}")
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
    async_client = httpx.AsyncClient(timeout=30.0)

    # Get tempo
    tempo = subtensor.get_subnet_hyperparameters(netuid).tempo
    bt.logging.info(f"Subnet tempo: {tempo} blocks")

    last_weight_block = 0
    scores = np.zeros(runtime.metagraph.n)

    try:
        while not stop_event.is_set():
            current_block = subtensor.get_current_block()
            last_heartbeat[0] = time.time()

            # Sync metagraph (periodically)
            if current_block % 100 == 0:
                runtime.metagraph.sync(subtensor=subtensor)
                # Resize scores if metagraph grew
                if len(scores) < runtime.metagraph.n:
                    new_scores = np.zeros(runtime.metagraph.n)
                    new_scores[: len(scores)] = scores
                    scores = new_scores

            # --- 1. Run Validation Step ---
            step_scores = await forward_module.run_step(
                task_gen, runtime.metagraph, runtime.wallet, async_client
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

                # Normalize scores to weights
                if np.sum(scores) > 0:
                    weights = scores / np.sum(scores)
                else:
                    weights = np.zeros_like(scores)

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
        "--enable-validator-api",
        action="store_true",
        default=os.getenv("ENABLE_VALIDATOR_API", "0") in {"1", "true", "True"},
        help="Enable the optional public validator API with /search and /health endpoints.",
    )
    parser.add_argument(
        "--validator-api-host",
        type=str,
        default=os.getenv("VALIDATOR_API_HOST", "0.0.0.0"),
        help="Host for the optional validator API.",
    )
    parser.add_argument(
        "--validator-api-port",
        type=int,
        default=int(os.getenv("VALIDATOR_API_PORT", "8010")),
        help="Port for the optional validator API.",
    )
    parser.add_argument(
        "--validator-api-max-miners",
        type=int,
        default=int(os.getenv("VALIDATOR_API_MAX_MINERS", "3")),
        help="Maximum number of miners the optional validator API will query per request.",
    )
    parser.add_argument(
        "--validator-api-miner-timeout-seconds",
        type=float,
        default=float(os.getenv("VALIDATOR_API_MINER_TIMEOUT_SECONDS", "60")),
        help="Per-miner timeout in seconds for optional validator API search requests.",
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

        runtime = ValidatorGatewayRuntime(
            wallet=wallet,
            metagraph=metagraph,
            scores=np.zeros(metagraph.n),
            score_lock=threading.Lock(),
            max_miners_per_request=max(1, int(config.validator_api_max_miners)),
            miner_request_timeout_seconds=max(
                1.0, float(config.validator_api_miner_timeout_seconds)
            ),
        )

        if config.enable_validator_api:
            import uvicorn

            gateway_app = create_validator_gateway(runtime)

            def run_gateway():
                uvicorn.run(
                    gateway_app,
                    host=config.validator_api_host,
                    port=config.validator_api_port,
                )

            api_thread = threading.Thread(target=run_gateway, daemon=True)
            api_thread.start()
            bt.logging.info(
                f"Validator API enabled on {config.validator_api_host}:{config.validator_api_port}"
            )

        bt.logging.info("Starting validator loop...")
        asyncio.run(
            run_validator_loop(
                subtensor, runtime, config.netuid, stop_event, last_heartbeat, config
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
