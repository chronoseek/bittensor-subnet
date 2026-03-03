import httpx
import time
import logging
import bittensor as bt
from typing import List, Tuple
from chronoseek.schemas import VideoSearchRequest, VideoSearchResponse
from chronoseek.scoring import score_response
from chronoseek.epistula import generate_header

logger = logging.getLogger(__name__)

async def query_miner(
    client: httpx.AsyncClient, 
    endpoint: str, 
    request: VideoSearchRequest,
    wallet: bt.Wallet
) -> Tuple[VideoSearchResponse, float]:
    """
    Query a single miner with Epistula signing.
    Returns (Response, Latency).
    """
    start_time = time.time()
    try:
        # Ensure endpoint has scheme
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
            
        # Generate Epistula headers
        headers = generate_header(wallet.hotkey, request.model_dump())
        
        # MVP: Increase timeout to 60s because miners download video on the fly
        resp = await client.post(
            f"{endpoint}/search", 
            json=request.model_dump(),
            headers=headers,
            timeout=60.0 
        )
        resp.raise_for_status()
        latency = time.time() - start_time
        return VideoSearchResponse(**resp.json()), latency
        
    except Exception as e:
        logger.debug(f"Failed to query miner {endpoint}: {e}")
        return VideoSearchResponse(results=[]), 0.0

async def run_step(
    task_gen,
    metagraph: bt.Metagraph,
    wallet: bt.Wallet,
    client: httpx.AsyncClient
) -> List[Tuple[int, float]]:
    """
    Run a single validation step:
    1. Generate task (ActivityNet)
    2. Query all miners via HTTP + Epistula
    3. Score responses (Strict IoU)
    
    Returns: List of (uid, score)
    """
    
    # 1. Generate Task
    video_url, query, ground_truth = task_gen.generate_task()
    bt.logging.info(f"Generated task: '{query}' for {video_url} | GT: {ground_truth}")
    
    request_model = VideoSearchRequest(video_url=video_url, query=query)
    
    scores = []
    
    # MVP: Loop over metagraph to query miners
    # We skip UIDs with no IP (0.0.0.0) or private IPs if not local dev
    for uid in metagraph.uids:
        axon = metagraph.axons[uid]
        if axon.ip == "0.0.0.0": 
            continue
        
        endpoint = f"{axon.ip}:{axon.port}"
        bt.logging.debug(f"Querying miner {uid} at {endpoint}...")
        
        resp, latency = await query_miner(client, endpoint, request_model, wallet)
        
        if not resp.results:
            bt.logging.debug(f"Miner {uid} returned no results.")
            score = 0.0
        else:
            score = score_response(resp.results, ground_truth, latency)
            bt.logging.info(f"Miner {uid} response: {resp.results[0]} | Score: {score}")
            
        scores.append((uid, score))
    
    return scores
