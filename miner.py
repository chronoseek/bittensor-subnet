"""
ChronoSeek Miner.
Exposes the miner logic via a FastAPI endpoint with Epistula verification.
"""

import os
import argparse
import uvicorn
import bittensor as bt
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chronoseek.protocol_models import (
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)
from chronoseek.miner.auth import ValidatorAuthContext, authorize_hotkey
from chronoseek.miner import logic as miner_logic_module
from chronoseek.epistula import verify_signature

app = FastAPI()
# Global logic instance (initialized in main)
miner_logic = None
validator_auth = None


def build_protocol_error(
    *,
    code: str,
    message: str,
    status_code: int,
    request_id: str | None = None,
    details: dict | None = None,
):
    payload = ProtocolError(
        error={
            "code": code,
            "message": message,
            "details": {
                **(details or {}),
                **({"request_id": request_id} if request_id else {}),
            }
            or None,
        }
    )
    return JSONResponse(
        status_code=status_code, content=payload.model_dump(mode="json")
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
):
    request_id = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            request_id = body.get("request_id")
    except Exception:
        request_id = None

    return build_protocol_error(
        code="INVALID_REQUEST",
        message="The search request payload is invalid.",
        status_code=400,
        request_id=request_id,
        details={"errors": exc.errors()},
    )


@app.post("/search", response_model=VideoSearchResponse)
async def search(
    request: Request,
    payload: VideoSearchRequest,
    caller_hotkey: str = Depends(verify_signature),
):
    """
    Handle search requests from validators.
    The verify_signature dependency ensures the request is authenticated.
    """
    bt.logging.info(
        f"Received request {payload.request_id or 'unknown-request'} from {caller_hotkey}: {payload.query}"
    )
    bt.logging.debug(f"Video URL: {payload.video_url}")

    if miner_logic is None:
        bt.logging.error("Miner logic not initialized")
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="Miner logic not initialized.",
            status_code=503,
            request_id=payload.request_id,
        )

    if payload.protocol_version != "2026-03-01":
        return build_protocol_error(
            code="UNSUPPORTED_PROTOCOL_VERSION",
            message="The miner does not support this protocol version.",
            status_code=400,
            request_id=payload.request_id,
        )

    is_authorized, auth_details = authorize_hotkey(validator_auth, caller_hotkey)
    if not is_authorized:
        bt.logging.warning(
            f"Rejecting request {payload.request_id or 'unknown-request'} from hotkey {caller_hotkey} with stake {auth_details['caller_stake']:.6f} below minimum {auth_details['minimum_validator_stake']:.6f}"
        )
        return build_protocol_error(
            code="INVALID_REQUEST",
            message="Caller hotkey does not meet the minimum validator stake requirement.",
            status_code=403,
            request_id=payload.request_id,
            details=auth_details,
        )

    try:
        bt.logging.info("Starting search processing...")
        results = miner_logic.search(
            payload.video_url, payload.query, top_k=payload.top_k
        )
        bt.logging.success(f"Search completed. Found {len(results)} results.")
        return VideoSearchResponse(request_id=payload.request_id, results=results)
    except miner_logic_module.SearchPipelineError as e:
        bt.logging.error(
            f"Request {payload.request_id or 'unknown-request'} failed with {e.code}: {e.message}"
        )
        status_code = 500
        if e.code in {
            "INVALID_REQUEST",
            "UNSUPPORTED_PROTOCOL_VERSION",
            "QUERY_INVALID",
        }:
            status_code = 400
        elif e.code == "VIDEO_FETCH_FAILED":
            status_code = 502
        elif e.code == "VIDEO_UNREADABLE":
            status_code = 422
        elif e.code == "TIMEOUT":
            status_code = 504

        return build_protocol_error(
            code=e.code,
            message=e.message,
            status_code=status_code,
            request_id=payload.request_id,
            details=e.details,
        )
    except Exception as e:
        bt.logging.error(f"Error processing request: {e}")
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="The miner encountered an unexpected internal error.",
            status_code=500,
            request_id=payload.request_id,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


def get_config():
    """
    Parse arguments and return configuration.
    Priority: CLI > Environment Variables > Defaults
    """
    parser = argparse.ArgumentParser(description="ChronoSeek Miner")

    # Add bittensor arguments first
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Axon.add_args(parser)
    bt.logging.add_args(parser)

    # Add custom arguments
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", "1")),
        help="Subnet NetUID",
    )
    parser.add_argument(
        "--min-validator-stake",
        type=float,
        default=float(os.getenv("MIN_VALIDATOR_STAKE", "10000")),
        help="Minimum validator stake required for the miner to accept a signed request.",
    )

    # Set defaults from environment variables for bittensor arguments
    defaults = {
        "wallet.name": os.getenv("WALLET_NAME", "default"),
        "wallet.hotkey": os.getenv("HOTKEY_NAME", "default"),
        "wallet.path": os.getenv("WALLET_PATH", "~/.bittensor/wallets/"),
        "subtensor.network": os.getenv("NETWORK", "finney"),
        "axon.port": int(os.getenv("PORT", "8000")),
        "logging.level": os.getenv("LOG_LEVEL", "INFO"),
    }
    parser.set_defaults(**defaults)

    return bt.Config(parser)


def resolve_server_port(config) -> int:
    axon_port = getattr(getattr(config, "axon", None), "port", None)
    if isinstance(axon_port, int) and not isinstance(axon_port, bool):
        return axon_port

    for index, arg in enumerate(os.sys.argv):
        if arg == "--axon.port" and index + 1 < len(os.sys.argv):
            try:
                return int(os.sys.argv[index + 1])
            except ValueError:
                break

    raise ValueError(
        "Miner server port is not configured. Set --axon.port or provide PORT before config parsing."
    )

def main():
    global miner_logic
    global validator_auth

    # 0. Load configuration
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
        f"Starting ChronoSeek Miner on network={config.subtensor.network}, netuid={config.netuid}"
    )
    bt.logging.info(f"Full config: {config}")

    # 1. Setup Bittensor objects
    wallet = bt.Wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    subtensor = bt.Subtensor(config=config)
    metagraph = bt.Metagraph(netuid=config.netuid, network=subtensor.network)

    # 2. Check Registration
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Miner hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}"
        )
        # return # Commented out for local testing if needed, but in prod we should return

    if wallet.hotkey.ss58_address in metagraph.hotkeys:
        bt.logging.info(
            f"Miner registered with UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}"
        )

    validator_auth = ValidatorAuthContext(
        min_validator_stake=max(0.0, float(config.min_validator_stake)),
        metagraph=metagraph,
    )
    bt.logging.info(
        f"Configured minimum validator stake requirement: {validator_auth.min_validator_stake:.6f}"
    )

    # Initialize Logic
    miner_logic = miner_logic_module.MinerLogic()

    # Determine port
    server_port = resolve_server_port(config)

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
    try:
        uvicorn.run(app, host="0.0.0.0", port=server_port)
    except KeyboardInterrupt:
        bt.logging.info("Miner stopped by user")
    except Exception as e:
        bt.logging.error(f"Miner error: {e}")


if __name__ == "__main__":
    main()
