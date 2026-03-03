"""
Semantic Video Moment Retrieval (SVMR) Subnet Validator.
Implements the synthetic task generation, miner querying (HTTP+Epistula), and IoU scoring loop.
"""

import os
import time
import argparse
import asyncio
import httpx
import bittensor as bt
import threading
import sys
import numpy as np
from typing import List

# Use ActivityNetTaskGenerator for MVP
from chronoseek.validator.task_gen import ActivityNetTaskGenerator
from chronoseek.validator.forward import run_step

HEARTBEAT_TIMEOUT = 600  # seconds

def heartbeat_monitor(last_heartbeat, stop_event):
    while not stop_event.is_set():
        time.sleep(5)
        if time.time() - last_heartbeat[0] > HEARTBEAT_TIMEOUT:
            bt.logging.error("No heartbeat detected in the last 600 seconds. Restarting process.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

async def run_validator_loop(
    subtensor: bt.Subtensor,
    wallet: bt.Wallet,
    metagraph: bt.Metagraph,
    netuid: int,
    stop_event: threading.Event,
    last_heartbeat: List[float]
):
    """
    Async validator loop.
    """
    # Initialize components
    task_gen = ActivityNetTaskGenerator()
    async_client = httpx.AsyncClient(timeout=30.0)
    
    # Get tempo
    tempo = subtensor.get_subnet_hyperparameters(netuid).tempo
    bt.logging.info(f"Subnet tempo: {tempo} blocks")
    
    last_weight_block = 0
    scores = np.zeros(metagraph.n)
    
    try:
        while not stop_event.is_set():
            current_block = subtensor.get_current_block()
            last_heartbeat[0] = time.time()
            
            # Sync metagraph (periodically)
            if current_block % 100 == 0:
                metagraph.sync(subtensor=subtensor)
                # Resize scores if metagraph grew
                if len(scores) < metagraph.n:
                    new_scores = np.zeros(metagraph.n)
                    new_scores[:len(scores)] = scores
                    scores = new_scores
            
            # --- 1. Run Validation Step ---
            step_scores = await run_step(task_gen, metagraph, wallet, async_client)
            
            # Update moving average scores
            alpha = 0.1
            for uid, score in step_scores:
                if uid < len(scores):
                    scores[uid] = alpha * score + (1 - alpha) * scores[uid]
            
            # --- 2. Set Weights ---
            blocks_since_last = current_block - last_weight_block
            if blocks_since_last >= tempo:
                bt.logging.info(f"Block {current_block}: Setting weights (tempo={tempo})")
                
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
                        wallet=wallet,
                        netuid=netuid,
                        uids=uids_list,
                        weights=weights_list,
                        wait_for_inclusion=True,
                        wait_for_finalization=False,
                    )
                    if success:
                        bt.logging.info("Successfully set weights")
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
    
    # Wallet args
    parser.add_argument("--wallet.name", default=os.getenv("WALLET_NAME", "default"), help="Wallet name")
    parser.add_argument("--wallet.hotkey", default=os.getenv("HOTKEY_NAME", "default"), help="Hotkey name")
    
    # Subtensor args
    parser.add_argument("--netuid", type=int, default=int(os.getenv("NETUID", "1")), help="Subnet NetUID")
    parser.add_argument("--subtensor.network", default=os.getenv("NETWORK", "finney"), help="Bittensor network")
    parser.add_argument("--subtensor.chain_endpoint", default=None, help="Chain endpoint")

    # Logging
    parser.add_argument("--logging.level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "TRACE"], help="Logging level")
    
    # Bittensor CLI config
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)

    return bt.Config(parser)

def main():
    """Run the SVMR subnet validator."""
    # 1. Configuration
    config = get_config()
    
    # Setup logging
    bt.logging(config=config, logging_dir=config.logging.logging_dir)
    if config.logging.level == "DEBUG":
        bt.logging.set_debug(True)
    elif config.logging.level == "TRACE":
        bt.logging.set_trace(True)
    
    bt.logging.info(f"Starting SVMR Validator on network={config.subtensor.network}, netuid={config.netuid}")

    # Heartbeat setup
    last_heartbeat = [time.time()]
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, args=(last_heartbeat, stop_event), daemon=True)
    heartbeat_thread.start()

    try:
        # 2. Setup
        wallet = bt.Wallet(config=config)
        bt.logging.info(f"Wallet: {wallet}")

        try:
            if wallet.hotkey:
                bt.logging.info(f"Starting Validator with hotkey: {wallet.hotkey.ss58_address}")
        except Exception as e:
            bt.logging.error(f"Error checking wallet: {e}")
            stop_event.set()
            return

        subtensor = bt.Subtensor(config=config)
        metagraph = bt.Metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
        metagraph.sync(subtensor=subtensor)
        
        # 3. Check Registration
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            bt.logging.error(f"Validator hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}")
            stop_event.set()
            return
            
        bt.logging.info(f"Validator registered with UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}")
        
        asyncio.run(run_validator_loop(
            subtensor, wallet, metagraph, config.netuid, stop_event, last_heartbeat
        ))
        
    except KeyboardInterrupt:
        bt.logging.info("Validator stopped by user")
    except Exception as e:
        bt.logging.error(f"Fatal error: {e}")
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)

if __name__ == "__main__":
    main()
