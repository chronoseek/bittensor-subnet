"""
ChronoSeek Chutes runtime.

This module exposes the HTTP contract validators query on the miner's deployed
Chutes runtime. The subnet miner command does not serve this app locally.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import bittensor as bt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from chronoseek.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MIN_VALIDATOR_STAKE,
    DEFAULT_NETUID,
    DEFAULT_NETWORK,
)
from chronoseek.bittensor_sdk import fetch_metagraph
from chronoseek.config import PROTOCOL_VERSION
from chronoseek.epistula import verify_signature
from chronoseek.logging import configure_logging, logger
from chronoseek.miner import logic as miner_logic_module
from chronoseek.miner.auth import ValidatorAuthContext, authorize_hotkey
from chronoseek.miner.cookie_refresh import (
    start_periodic_cookie_refresh,
    stop_periodic_cookie_refresh,
)
from chronoseek.protocol_models import (
    ProofOfAccessRequest,
    ProofOfAccessResponse,
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)
from chronoseek.video.downloader import VideoDownloader, VideoDownloadError
from chronoseek.video.proof_of_access import compute_proof_of_access_hash

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

# execute_search/execute_proof_of_access run a real video download plus (for
# search) GPU inference - both synchronous and slow. Run them off the event
# loop entirely so /health stays responsive to Chutes' verification probe
# while one is in flight, and serialize them so only one runs at a time,
# matching the implicit one-at-a-time access the blocking calls used to give
# the shared CLIP model/GPU for free.
_inference_semaphore = asyncio.Semaphore(1)


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        netuid=int(os.getenv("NETUID", str(DEFAULT_NETUID))),
        network=os.getenv("NETWORK", DEFAULT_NETWORK),
        min_validator_stake=float(
            os.getenv("MIN_VALIDATOR_STAKE", str(DEFAULT_MIN_VALIDATOR_STAKE))
        ),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
    )


def configure_runtime_logging(config: RuntimeConfig) -> None:
    configure_logging(config.log_level)


def load_runtime_metagraph(config: RuntimeConfig):
    subtensor = bt.Subtensor(network=config.network)
    return fetch_metagraph(subtensor, config.netuid)


def initialize_runtime() -> None:
    global miner_logic
    global validator_auth
    global startup_error

    config = load_runtime_config()
    configure_runtime_logging(config)
    logger.info(
        f"Starting ChronoSeek Chutes runtime on network={config.network}, netuid={config.netuid}"
    )
    start_periodic_cookie_refresh()

    try:
        metagraph = load_runtime_metagraph(config)
        validator_auth = ValidatorAuthContext(
            min_validator_stake=max(0.0, float(config.min_validator_stake)),
            metagraph=metagraph,
        )
        miner_logic = miner_logic_module.MinerLogic()
        startup_error = None
        logger.success("ChronoSeek Chutes runtime initialized.")
    except Exception as exc:
        miner_logic = None
        validator_auth = None
        startup_error = str(exc)
        logger.error(f"ChronoSeek Chutes runtime initialization failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_runtime()
    try:
        yield
    finally:
        stop_periodic_cookie_refresh()


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
) -> dict | JSONResponse:
    request_id = payload.request_id or "unknown-request"
    caller = caller_hotkey or "chutes-sdk-authenticated-caller"
    logger.info(f"Received request {request_id} from {caller}: {payload.query}")
    logger.debug(f"Video URL: {payload.video_url}")

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
            logger.warning(
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
        logger.info("Starting search processing...")
        results = miner_logic.search(payload.video_url, payload.query, top_k=payload.top_k)
        logger.success(f"Search completed. Found {len(results)} results.")
        response = VideoSearchResponse(
            request_id=payload.request_id,
            results=results,
        )
        return response.model_dump(mode="json")
    except miner_logic_module.SearchPipelineError as exc:
        logger.error(f"Request {request_id} failed with {exc.code}: {exc.message}")
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
        logger.error(f"Error processing request: {exc}")
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="The runtime encountered an unexpected internal error.",
            status_code=500,
            request_id=payload.request_id,
        )


def execute_proof_of_access(
    payload: ProofOfAccessRequest,
    *,
    caller_hotkey: str | None = None,
    enforce_validator_auth: bool = True,
) -> dict | JSONResponse:
    request_id = payload.request_id or "unknown-request"
    caller = caller_hotkey or "chutes-sdk-authenticated-caller"
    logger.info(f"Received proof-of-access request {request_id} from {caller}: {payload.video_url}")

    if validator_auth is None:
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
            logger.warning(
                f"Rejecting proof-of-access request {request_id} from hotkey {caller_hotkey} with stake {auth_details['caller_stake']:.6f} below minimum {auth_details['minimum_validator_stake']:.6f}"
            )
            return build_protocol_error(
                code="INVALID_REQUEST",
                message="Caller hotkey does not meet the minimum validator stake requirement.",
                status_code=403,
                request_id=payload.request_id,
                details=auth_details,
            )

    downloaded_video = None
    try:
        downloaded_video = VideoDownloader.download_video(
            payload.video_url,
            raise_on_failure=True,
        )
    except VideoDownloadError as exc:
        logger.error(f"Proof-of-access download failed: {exc}")
        return build_protocol_error(
            code="VIDEO_FETCH_FAILED",
            message="The video URL could not be fetched by supported download backends.",
            status_code=502,
            request_id=payload.request_id,
            details=exc.details(),
        )

    try:
        content_hash = compute_proof_of_access_hash(downloaded_video.path)
        if content_hash is None:
            return build_protocol_error(
                code="VIDEO_UNREADABLE",
                message="The downloaded video could not be decoded into frames.",
                status_code=422,
                request_id=payload.request_id,
            )

        response = ProofOfAccessResponse(
            request_id=payload.request_id,
            content_hash=content_hash,
        )
        return response.model_dump(mode="json")
    except Exception as exc:
        logger.error(f"Error processing proof-of-access request: {exc}")
        return build_protocol_error(
            code="INTERNAL_ERROR",
            message="The runtime encountered an unexpected internal error.",
            status_code=500,
            request_id=payload.request_id,
        )
    finally:
        VideoDownloader.cleanup(downloaded_video)


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
    async with _inference_semaphore:
        return await asyncio.to_thread(
            execute_search,
            payload,
            caller_hotkey=caller_hotkey,
            enforce_validator_auth=True,
        )


@app.post("/proof-of-access", response_model=ProofOfAccessResponse)
async def proof_of_access(
    payload: ProofOfAccessRequest,
    caller_hotkey: str = Depends(verify_signature),
):
    async with _inference_semaphore:
        return await asyncio.to_thread(
            execute_proof_of_access,
            payload,
            caller_hotkey=caller_hotkey,
            enforce_validator_auth=True,
        )
