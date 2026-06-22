import asyncio
import json
import time
from dataclasses import dataclass
from typing import List, Tuple
from uuid import uuid4
from urllib.parse import urlparse

import bittensor as bt
import httpx

from chronoseek.protocol_models import (
    ProtocolError,
    VideoSearchRequest,
    VideoSearchResponse,
)
from chronoseek.scoring import score_response
from chronoseek.epistula import generate_header
from chronoseek.chutes.runtime import ChutesRuntimeEndpoint
from chronoseek.validator.task_models import normalize_generated_task

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


def _normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith("http"):
        return endpoint
    return f"http://{endpoint}"


async def query_miner(
    client: httpx.AsyncClient,
    uid: int,
    hotkey: str,
    endpoint: str,
    request: VideoSearchRequest,
    wallet: bt.Wallet,
    timeout_seconds: float = 60.0,
    extra_headers: dict[str, str] | None = None,
) -> MinerQueryResult:
    """
    Query one resolved miner runtime with Epistula signing.
    """
    start_time = time.time()
    try:
        request_payload = request.model_dump(mode="json")
        endpoint = _normalize_endpoint(endpoint)

        # Generate Epistula headers. Provider auth stays additive.
        headers = {
            **(extra_headers or {}),
            **generate_header(wallet.hotkey, request_payload),
        }

        resp = await client.post(
            f"{endpoint}/search",
            json=request_payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        latency = time.time() - start_time
        bt.logging.info(f"Query miner {uid} ({hotkey}, {endpoint}) response: {resp.json()}")
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

        bt.logging.warning(
            f"Failed to query miner {uid} ({hotkey}, {endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except httpx.TimeoutException as exc:
        failure = MinerQueryFailure(kind="timeout", message=str(exc))
        bt.logging.warning(
            f"Failed to query miner {uid} ({hotkey}, {endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except httpx.ConnectError as exc:
        failure = MinerQueryFailure(kind="connect_error", message=str(exc))
        bt.logging.warning(
            f"Failed to query miner {uid} ({hotkey}, {endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )
    except Exception as exc:
        failure = MinerQueryFailure(kind="unexpected_error", message=str(exc))
        bt.logging.warning(
            f"Failed to query miner {uid} ({hotkey}, {endpoint}): {failure.message}"
        )
        return MinerQueryResult(
            response=VideoSearchResponse(results=[]),
            latency=0.0,
            failure=failure,
        )


async def query_uid(
    semaphore: asyncio.Semaphore,
    miner_endpoint: ChutesRuntimeEndpoint,
    client: httpx.AsyncClient,
    request_model: VideoSearchRequest,
    wallet: bt.Wallet,
    ground_truths: List[Tuple[float, float]],
    timeout_seconds: float,
    clip_duration: float | None = None,
    max_prediction_duration_seconds: float | None = None,
    extra_headers: dict[str, str] | None = None,
    expects_empty_response: bool = False,
) -> Tuple[int, float]:
    async with semaphore:
        uid = miner_endpoint.uid
        bt.logging.debug(
            f"Querying miner {uid} runtime from chain submission at {miner_endpoint.endpoint}..."
        )
        result = await query_miner(
            client,
            uid,
            miner_endpoint.hotkey,
            miner_endpoint.endpoint,
            request_model,
            wallet,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
        )
        resp = result.response
        latency = result.latency

        if not resp.results:
            if expects_empty_response and result.failure is None:
                bt.logging.success(
                    f"[UID {uid}] Empty canary response accepted | "
                    f"Request: {request_model.request_id} | "
                    f"Latency: {latency:.2f}s | Score: 1.0000"
                )
                return int(uid), 1.0

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

        score = score_response(
            resp.results,
            ground_truths,
            latency,
            clip_duration=clip_duration,
            max_prediction_duration_seconds=max_prediction_duration_seconds,
            score_top_k=1,
            expects_empty_response=expects_empty_response,
        )
        result = resp.results[0]
        res_str = f"[{result.start:.1f}s - {result.end:.1f}s]"
        bt.logging.success(
            f"[UID {uid}] Request: {request_model.request_id} | Score: {score:.4f} | Latency: {latency:.2f}s | Result: {res_str}"
        )
        return int(uid), score


async def run_step(
    task_gen,
    metagraph: bt.Metagraph,
    wallet: bt.Wallet,
    client: httpx.AsyncClient,
    miner_timeout_seconds: float = 60.0,
    miner_endpoints: list[ChutesRuntimeEndpoint] | None = None,
    provider_headers: dict[str, str] | None = None,
    validator_eval_top_k: int = 1,
    max_prediction_duration_seconds: float = 60.0,
) -> List[Tuple[int, float]]:
    """
    Run a single validation step:
    1. Generate task (ActivityNet)
    2. Query selected responsive runtime endpoints via HTTP + Epistula
    3. Score responses with continuous IoU

    Returns: List of (uid, score)
    """

    # 1. Generate Task
    bt.logging.info("=" * 50)
    bt.logging.info(f"STARTING VALIDATION STEP")
    bt.logging.info("=" * 50)

    bt.logging.info(">>> Phase 1: Task Generation (ActivityNet)")
    try:
        normalized_task = normalize_generated_task(
            task_gen.generate_task(),
            default_top_k=validator_eval_top_k,
        )
    except RuntimeError as exc:
        bt.logging.warning(
            f"Task generation could not find an accessible video: {exc}. Refreshing video availability checks and retrying."
        )
        refreshed_entries = 0
        refresh_lookup = getattr(task_gen, "refresh_video_lookup", None)
        if callable(refresh_lookup):
            refreshed_entries = int(refresh_lookup())
            bt.logging.info(
                f"Refreshed {refreshed_entries} cached unavailable video availability entries."
            )
        try:
            normalized_task = normalize_generated_task(
                task_gen.generate_task(),
                default_top_k=validator_eval_top_k,
            )
        except RuntimeError as retry_exc:
            bt.logging.warning(
                f"Skipping validation step because no accessible validator task was found after retry: {retry_exc}"
            )
            bt.logging.info("=" * 50)
            return []
    request_id = normalized_task.request_id or f"validation-{uuid4()}"
    video_url = normalized_task.video_url
    query = normalized_task.query
    ground_truths = normalized_task.ground_truths
    clip_duration = normalized_task.clip_duration
    artifact_host = urlparse(video_url).netloc or "unknown"

    bt.logging.info("-" * 40)
    bt.logging.info(f"Request ID:  {request_id}")
    bt.logging.info(f"Task ID:     {normalized_task.task_id or 'legacy-task'}")
    bt.logging.info(f"Artifact Host: {artifact_host}")
    bt.logging.info(f"Query Variant: {normalized_task.query_variant_id or 'legacy-query'}")
    if normalized_task.canary_kind:
        bt.logging.info(
            f"Canary: {normalized_task.canary_kind} | "
            f"expects_empty={normalized_task.expects_empty_response}"
        )
    bt.logging.info(f"Clip Duration: {clip_duration if clip_duration is not None else 'unknown'}")
    bt.logging.info(f"Ground Truth Count: {len(ground_truths)}")
    bt.logging.info("-" * 40)

    request_model = VideoSearchRequest(
        request_id=request_id,
        video={"url": video_url},
        query=query,
        top_k=normalized_task.top_k,
    )

    scores = []

    miner_endpoints = list(miner_endpoints or [])
    bt.logging.info(
        "\n>>> Phase 2: Querying Miners "
        f"({len(metagraph.uids)} total | selected={len(miner_endpoints)} | "
        f"routing=chain+chutes)"
    )

    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MINER_REQUESTS)

    for miner_endpoint in miner_endpoints:
        tasks.append(
            query_uid(
                semaphore,
                miner_endpoint,
                client,
                request_model,
                wallet,
                ground_truths,
                miner_timeout_seconds,
                clip_duration=clip_duration,
                max_prediction_duration_seconds=max_prediction_duration_seconds,
                extra_headers=provider_headers,
                expects_empty_response=normalized_task.expects_empty_response,
            )
        )

    if not tasks:
        bt.logging.warning("No eligible miners were selected for this validation step.")
        bt.logging.info("=" * 50)
        return []

    scores.extend(await asyncio.gather(*tasks))

    bt.logging.info("=" * 50)
    return scores
