"""
ChronoSeek Chutes runtime.

This module exposes the HTTP contract validators query on the miner's deployed
Chutes runtime. The subnet miner command does not serve this app locally.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import bittensor as bt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from chronoseek.config import PROTOCOL_VERSION
from chronoseek.epistula import verify_signature
from chronoseek.miner import logic as miner_logic_module
from chronoseek.miner.auth import ValidatorAuthContext, authorize_hotkey
from chronoseek.protocol_models import (
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)

load_dotenv()


@dataclass(frozen=True)
class RuntimeConfig:
    netuid: int
    network: str
    min_validator_stake: float
    log_level: str


miner_logic = None
validator_auth = None
startup_error: str | None = None


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        netuid=int(os.getenv("NETUID", "1")),
        network=os.getenv("NETWORK", "finney"),
        min_validator_stake=float(os.getenv("MIN_VALIDATOR_STAKE", "10000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def configure_runtime_logging(config: RuntimeConfig) -> None:
    bt.logging.on()
    if config.log_level == "DEBUG":
        bt.logging.set_debug(True)
    elif config.log_level == "TRACE":
        bt.logging.set_trace(True)
    else:
        bt.logging.set_info(True)


def load_runtime_metagraph(config: RuntimeConfig):
    subtensor = bt.Subtensor(network=config.network)
    metagraph = bt.Metagraph(
        netuid=config.netuid,
        network=subtensor.network,
        sync=False,
    )
    metagraph.sync(subtensor=subtensor)
    return metagraph


def initialize_runtime() -> None:
    global miner_logic
    global validator_auth
    global startup_error

    config = load_runtime_config()
    configure_runtime_logging(config)
    bt.logging.info(
        f"Starting ChronoSeek Chutes runtime on network={config.network}, netuid={config.netuid}"
    )

    try:
        metagraph = load_runtime_metagraph(config)
        validator_auth = ValidatorAuthContext(
            min_validator_stake=max(0.0, float(config.min_validator_stake)),
            metagraph=metagraph,
        )
        miner_logic = miner_logic_module.MinerLogic()
        startup_error = None
        bt.logging.success("ChronoSeek Chutes runtime initialized.")
    except Exception as exc:
        miner_logic = None
        validator_auth = None
        startup_error = str(exc)
        bt.logging.error(f"ChronoSeek Chutes runtime initialization failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_runtime()
    yield


app = FastAPI(lifespan=lifespan)


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
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


def health_payload() -> dict:
    ready = miner_logic is not None and validator_auth is not None
    return {
        "ok": ready,
        "status": "ok" if ready else "not_ready",
        "service": "chronoseek-runtime",
        "protocol_versions": [PROTOCOL_VERSION],
        **({"error": startup_error} if startup_error else {}),
    }


def execute_search(
    payload: VideoSearchRequest,
    *,
    caller_hotkey: str | None = None,
    enforce_validator_auth: bool = True,
):
    request_id = payload.request_id or "unknown-request"
    caller = caller_hotkey or "chutes-sdk-authenticated-caller"
    bt.logging.info(f"Received request {request_id} from {caller}: {payload.query}")
    bt.logging.debug(f"Video URL: {payload.video_url}")

    if miner_logic is None or validator_auth is None:
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="Runtime is not initialized.",
            status_code=503,
            request_id=payload.request_id,
            details={"startup_error": startup_error} if startup_error else None,
        )

    if payload.protocol_version != PROTOCOL_VERSION:
        return build_protocol_error(
            code="UNSUPPORTED_PROTOCOL_VERSION",
            message="The runtime does not support this protocol version.",
            status_code=400,
            request_id=payload.request_id,
        )

    if enforce_validator_auth:
        if not caller_hotkey:
            return build_protocol_error(
                code="INVALID_REQUEST",
                message="Caller hotkey is required for validator authorization.",
                status_code=401,
                request_id=payload.request_id,
            )

        is_authorized, auth_details = authorize_hotkey(validator_auth, caller_hotkey)
        if not is_authorized:
            bt.logging.warning(
                f"Rejecting request {request_id} from hotkey {caller_hotkey} with stake {auth_details['caller_stake']:.6f} below minimum {auth_details['minimum_validator_stake']:.6f}"
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
        results = miner_logic.search(payload.video_url, payload.query, top_k=payload.top_k)
        bt.logging.success(f"Search completed. Found {len(results)} results.")
        return VideoSearchResponse(request_id=payload.request_id, results=results)
    except miner_logic_module.SearchPipelineError as exc:
        bt.logging.error(f"Request {request_id} failed with {exc.code}: {exc.message}")
        status_code = 500
        if exc.code in {
            "INVALID_REQUEST",
            "UNSUPPORTED_PROTOCOL_VERSION",
            "QUERY_INVALID",
        }:
            status_code = 400
        elif exc.code == "VIDEO_FETCH_FAILED":
            status_code = 502
        elif exc.code == "VIDEO_UNREADABLE":
            status_code = 422
        elif exc.code == "TIMEOUT":
            status_code = 504

        return build_protocol_error(
            code=exc.code,
            message=exc.message,
            status_code=status_code,
            request_id=payload.request_id,
            details=exc.details,
        )
    except Exception as exc:
        bt.logging.error(f"Error processing request: {exc}")
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="The runtime encountered an unexpected internal error.",
            status_code=500,
            request_id=payload.request_id,
        )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
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


@app.get("/health")
async def health():
    return health_payload()


@app.post("/search", response_model=VideoSearchResponse)
async def search(
    payload: VideoSearchRequest,
    caller_hotkey: str = Depends(verify_signature),
):
    return execute_search(
        payload,
        caller_hotkey=caller_hotkey,
        enforce_validator_auth=True,
    )
