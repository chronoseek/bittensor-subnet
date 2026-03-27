import httpx
import time
import logging
import asyncio
import json
from uuid import uuid4
import bittensor as bt
from dataclasses import dataclass
from typing import List, Tuple
from chronoseek.protocol_models import (
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)
from chronoseek.scoring import score_response
from chronoseek.epistula import generate_header

logger = logging.getLogger(__name__)
MAX_CONCURRENT_MINER_REQUESTS = 8


@dataclass
class MinerQueryFailure:
    kind: str
    message: str
    status_code: int | None = None
    protocol_code: str | None = None


@dataclass
class MinerQueryResult:
    response: VideoSearchResponse
    latency: float
    failure: MinerQueryFailure | None = None


async def query_miner(
    client: httpx.AsyncClient,
    endpoint: str,
    request: VideoSearchRequest,
    wallet: bt.Wallet,
    timeout_seconds: float = 60.0,
) -> MinerQueryResult:
    """
    Query a single miner with Epistula signing.
    Returns (Response, Latency).
    """
    start_time = time.time()
    try:
        request_payload = request.model_dump(mode="json")

        # Ensure endpoint has scheme
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"

        # Generate Epistula headers
        headers = generate_header(wallet.hotkey, request_payload)

        resp = await client.post(
            f"{endpoint}/search",
            json=request_payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        latency = time.time() - start_time
        return MinerQueryResult(
            response=VideoSearchResponse(**resp.json()), latency=latency
        )

    except httpx.HTTPStatusError as exc:
        failure = MinerQueryFailure(
            kind="http_status",
            message=str(exc),
            status_code=exc.response.status_code,
        )
        try:
            payload = ProtocolError(**exc.response.json())
            failure.protocol_code = payload.error.code
            failure.message = payload.error.message
        except (ValueError, json.JSONDecodeError, TypeError):
            pass

        logger.warning(
            f"Failed to query miner {wallet.hotkey.ss58_address} ({endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except httpx.TimeoutException as exc:
        failure = MinerQueryFailure(kind="timeout", message=str(exc))
        logger.warning(
            f"Failed to query miner {wallet.hotkey.ss58_address} ({endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except httpx.ConnectError as exc:
        failure = MinerQueryFailure(kind="connect_error", message=str(exc))
        logger.warning(
            f"Failed to query miner {wallet.hotkey.ss58_address} ({endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except Exception as exc:
        failure = MinerQueryFailure(kind="unexpected_error", message=str(exc))
        logger.warning(
            f"Failed to query miner {wallet.hotkey.ss58_address} ({endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )


async def query_uid(
    semaphore: asyncio.Semaphore,
    uid: int,
    endpoint: str,
    client: httpx.AsyncClient,
    request_model: VideoSearchRequest,
    wallet: bt.Wallet,
    ground_truths: List[Tuple[float, float]],
) -> Tuple[int, float]:
    async with semaphore:
        bt.logging.debug(f"Querying miner {uid} at {endpoint}...")
        result = await query_miner(client, endpoint, request_model, wallet)
        resp = result.response
        latency = result.latency

        if not resp.results:
            failure_suffix = ""
            if result.failure is not None:
                parts = [result.failure.kind]
                if result.failure.status_code is not None:
                    parts.append(str(result.failure.status_code))
                if result.failure.protocol_code:
                    parts.append(result.failure.protocol_code)
                if result.failure.message:
                    parts.append(result.failure.message)
                failure_suffix = " | Failure: " + " | ".join(parts)
            bt.logging.warning(
                f"[UID {uid}] No results | Request: {request_model.request_id} | Latency: {latency:.2f}s | Score: 0.0000{failure_suffix}"
            )
            return int(uid), 0.0

        score = score_response(resp.results, ground_truths, latency)
        result = resp.results[0]
        res_str = f"[{result.start:.1f}s - {result.end:.1f}s]"
        bt.logging.success(
            f"[UID {uid}] Request: {request_model.request_id} | Score: {score:.4f} | Latency: {latency:.2f}s | Result: {res_str}"
        )
        return int(uid), score


async def run_step(
    task_gen, metagraph: bt.Metagraph, wallet: bt.Wallet, client: httpx.AsyncClient
) -> List[Tuple[int, float]]:
    """
    Run a single validation step:
    1. Generate task (ActivityNet)
    2. Query all miners via HTTP + Epistula
    3. Score responses (Strict IoU)

    Returns: List of (uid, score)
    """

    # 1. Generate Task
    bt.logging.info("=" * 50)
    bt.logging.info(f"STARTING VALIDATION STEP")
    bt.logging.info("=" * 50)

    bt.logging.info(">>> Phase 1: Task Generation (ActivityNet)")
    video_url, query, ground_truths = task_gen.generate_task()
    request_id = f"validation-{uuid4()}"

    bt.logging.info("-" * 40)
    bt.logging.info(f"Request ID:  {request_id}")
    bt.logging.info(f"Video URL:   {video_url}")
    bt.logging.info(f"Query:       {query}")
    bt.logging.info(f"Ground Truths: {ground_truths}")
    bt.logging.info("-" * 40)

    request_model = VideoSearchRequest(
        request_id=request_id,
        video={"url": video_url},
        query=query,
    )

    scores = []

    # MVP: Loop over metagraph to query miners
    # We skip UIDs with no IP (0.0.0.0) or private IPs if not local dev
    bt.logging.info(f"\n>>> Phase 2: Querying Miners ({len(metagraph.uids)} total)")

    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MINER_REQUESTS)

    for uid in metagraph.uids:
        axon = metagraph.axons[uid]
        if axon.ip == "0.0.0.0":
            continue

        endpoint = f"http://{axon.ip}:{axon.port}"
        tasks.append(
            query_uid(
                semaphore,
                int(uid),
                endpoint,
                client,
                request_model,
                wallet,
                ground_truths,
            )
        )

    if tasks:
        scores.extend(await asyncio.gather(*tasks))

    bt.logging.info("=" * 50)
    return scores
