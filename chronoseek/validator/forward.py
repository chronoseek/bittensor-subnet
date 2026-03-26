import httpx
import time
import logging
import asyncio
from uuid import uuid4
import bittensor as bt
from typing import List, Tuple
from chronoseek.protocol_models import VideoSearchRequest, VideoSearchResponse
from chronoseek.scoring import score_response
from chronoseek.epistula import generate_header

logger = logging.getLogger(__name__)
MAX_CONCURRENT_MINER_REQUESTS = 8


async def query_miner(
    client: httpx.AsyncClient,
    endpoint: str,
    request: VideoSearchRequest,
    wallet: bt.Wallet,
) -> Tuple[VideoSearchResponse, float]:
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

        # MVP: Increase timeout to 60s because miners download video on the fly
        resp = await client.post(
            f"{endpoint}/search",
            json=request_payload,
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        latency = time.time() - start_time
        return VideoSearchResponse(**resp.json()), latency

    except Exception as e:
        logger.warning(f"Failed to query miner {endpoint}: {e}")
        return VideoSearchResponse(results=[]), 0.0


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
        resp, latency = await query_miner(client, endpoint, request_model, wallet)

        if not resp.results:
            bt.logging.warning(
                f"[UID {uid}] No results | Request: {request_model.request_id} | Latency: {latency:.2f}s | Score: 0.0000"
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
