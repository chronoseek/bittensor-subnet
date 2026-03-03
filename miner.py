"""
SVMR Subnet Miner.
Exposes the miner logic via a FastAPI endpoint with Epistula verification.
"""

import os
import argparse
import uvicorn
import bittensor as bt
from fastapi import FastAPI, Request, HTTPException, Depends
from chronoseek.schemas import VideoSearchRequest, VideoSearchResponse
from chronoseek.miner.logic import MinerLogic
from chronoseek.epistula import verify_signature

app = FastAPI()
# Global logic instance (initialized in main)
miner_logic = None

@app.post("/search", response_model=VideoSearchResponse)
async def search(
    request: Request, 
    payload: VideoSearchRequest,
    caller_hotkey: str = Depends(verify_signature)
):
    """
    Handle search requests from validators.
    The verify_signature dependency ensures the request is authenticated.
    """
    bt.logging.info(f"Received request from {caller_hotkey}: {payload.query}")
    
    if miner_logic is None:
         raise HTTPException(status_code=503, detail="Miner logic not initialized")

    try:
        # TODO: Check if caller_hotkey is a registered validator with stake
        results = miner_logic.search(payload.video_url, payload.query)
        return VideoSearchResponse(results=results)
    except Exception as e:
        bt.logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

def get_config():
    """
    Parse arguments and return configuration.
    Priority: CLI > Environment Variables > Defaults
    """
    parser = argparse.ArgumentParser(description="ChronoSeek Miner")
    
    # Wallet args
    parser.add_argument("--wallet.name", default=os.getenv("WALLET_NAME", "default"), help="Wallet name")
    parser.add_argument("--wallet.hotkey", default=os.getenv("HOTKEY_NAME", "default"), help="Hotkey name")
    
    # Subtensor args
    parser.add_argument("--netuid", type=int, default=int(os.getenv("NETUID", "1")), help="Subnet NetUID")
    parser.add_argument("--subtensor.network", default=os.getenv("NETWORK", "finney"), help="Bittensor network")
    parser.add_argument("--subtensor.chain_endpoint", default=None, help="Chain endpoint")

    # Miner args
    parser.add_argument("--axon.port", type=int, default=int(os.getenv("PORT", "8000")), help="Miner Axon Port")
    parser.add_argument("--logging.level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "TRACE"], help="Logging level")
    
    # Bittensor CLI config (to allow passing --wallet.path etc)
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)

    return bt.Config(parser)

def main():
    global miner_logic
    
    # 0. Load configuration
    config = get_config()
    
    # Setup logging
    bt.logging(config=config, logging_dir=config.logging.logging_dir)
    if config.logging.level == "DEBUG":
        bt.logging.set_debug(True)
    elif config.logging.level == "TRACE":
        bt.logging.set_trace(True)

    bt.logging.info(f"Starting SVMR Miner on network={config.subtensor.network}, netuid={config.netuid}")

    # 1. Setup Bittensor objects
    wallet = bt.Wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    subtensor = bt.Subtensor(config=config)
    metagraph = bt.Metagraph(netuid=config.netuid, network=subtensor.network)
    
    # 2. Check Registration
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"Miner hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}")
        # return # Commented out for local testing if needed, but in prod we should return

    if wallet.hotkey.ss58_address in metagraph.hotkeys:
        bt.logging.info(f"Miner registered with UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}")
    
    # Initialize Logic
    miner_logic = MinerLogic()
    
    # Determine port
    server_port = config.axon.port if config.is_set("axon.port") else config.port
    
    # 3. Serve Axon (Announce IP/Port to the network)
    bt.logging.info(f"Serving Axon on port {server_port}...")
    try:
        # Create axon object just for announcement (we run our own uvicorn)
        axon = bt.Axon(wallet=wallet, port=server_port)
        
        # Announce to the network
        bt.logging.info(f"Announcing axon to netuid {config.netuid}...")
        subtensor.serve_axon(
            netuid=config.netuid,
            axon=axon,
        )
        bt.logging.success(f"Served Axon successfully on port {server_port}")
        
    except Exception as e:
        bt.logging.error(f"Failed to serve Axon: {e}")
        # We continue even if serve fails, as it might just be a timeout or network issue
        # and the miner can still function if previously registered correctly.
    
    bt.logging.info(f"Starting Miner HTTP Server on port {server_port}")
    uvicorn.run(app, host="0.0.0.0", port=server_port)

if __name__ == "__main__":
    main()
