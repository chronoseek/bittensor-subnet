import asyncio
from dataclasses import dataclass, field
import json
from threading import Lock
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import bittensor as bt

from chronoseek.config import PROTOCOL_VERSION
from chronoseek.protocol_models import ProtocolError, VideoSearchRequest, VideoSearchResponse
from chronoseek.validator.forward import query_miner


@dataclass
class ValidatorGatewayRuntime:
    wallet: bt.Wallet
    metagraph: bt.Metagraph
    scores: np.ndarray
    score_lock: Lock
    max_miners_per_request: int
    metagraph_lock: Lock = field(default_factory=Lock)
    sync_miner_request_timeout_seconds: float = 60.0
    stream_miner_request_timeout_seconds: float = 60.0
    responsive_lock: Lock = field(default_factory=Lock)
    responsive_uids: set[int] = field(default_factory=set)
    responsive_initialized: bool = False
    responsive_last_refresh_at: float | None = None
    last_metagraph_sync_block: int | None = None


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


def _get_metagraph_snapshot(runtime: ValidatorGatewayRuntime):
    with runtime.metagraph_lock:
        return runtime.metagraph


def _rank_candidate_uids(runtime: ValidatorGatewayRuntime, metagraph: bt.Metagraph) -> list[int]:
    with runtime.score_lock:
        current_scores = np.array(runtime.scores, copy=True)
    with runtime.responsive_lock:
        responsive_uids = set(runtime.responsive_uids)
        responsive_initialized = runtime.responsive_initialized

    candidates: list[tuple[int, float]] = []
    for uid in metagraph.uids:
        uid = int(uid)
        axon = metagraph.axons[uid]
        if axon.ip == "0.0.0.0":
            continue
        if responsive_initialized and uid not in responsive_uids:
            continue
        score = float(current_scores[uid]) if uid < len(current_scores) else 0.0
        candidates.append((uid, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    ranked_uids = [uid for uid, _ in candidates]
    if runtime.max_miners_per_request > 0:
        ranked_uids = ranked_uids[: runtime.max_miners_per_request]
    return ranked_uids


def _dedupe_and_rank_results(results, top_k: int):
    ranked = sorted(results, key=lambda item: item.confidence, reverse=True)
    deduped = []
    seen = set()
    for result in ranked:
        key = (round(result.start, 3), round(result.end, 3), round(result.confidence, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
        if len(deduped) >= top_k:
            break
    return deduped


def _derive_gateway_failure(failures):
    protocol_codes = [
        failure.protocol_code for failure in failures if failure and failure.protocol_code
    ]
    unique_protocol_codes = set(protocol_codes)

    if len(unique_protocol_codes) == 1:
        protocol_code = next(iter(unique_protocol_codes))
        if protocol_code == "VIDEO_FETCH_FAILED":
            return (
                "VIDEO_FETCH_FAILED",
                "The video URL could not be downloaded. Check that the URL is valid and publicly accessible.",
                502,
            )
        if protocol_code == "VIDEO_UNREADABLE":
            return (
                "VIDEO_UNREADABLE",
                "The video was fetched but could not be decoded. Try a different video URL or file format.",
                422,
            )

    if failures and all(failure.kind == "timeout" for failure in failures):
        return ("TIMEOUT", "All miner queries timed out.", 504)

    return ("INTERNAL_ERROR", "No miner returned a usable search result.", 502)


def _build_completed_response(
    payload: VideoSearchRequest,
    ranked_uids: list[int],
    miner_results: list[tuple[int, Any]],
) -> VideoSearchResponse | None:
    aggregated_results = []
    successful_uids = []
    for uid, query_result in miner_results:
        if query_result.response.results:
            successful_uids.append(uid)
            aggregated_results.extend(query_result.response.results)

    if not aggregated_results:
        return None

    ranked_results = _dedupe_and_rank_results(aggregated_results, payload.top_k)
    return VideoSearchResponse(
        request_id=payload.request_id,
        status="completed",
        results=ranked_results,
        miner_metadata={
            "source": "validator-gateway",
            "selected_uids": successful_uids,
            "queried_uids": ranked_uids,
        },
    )


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _query_ranked_miners(
    runtime: ValidatorGatewayRuntime,
    metagraph: bt.Metagraph,
    payload: VideoSearchRequest,
    ranked_uids: list[int],
):
    async def run_query(uid: int):
        hotkeys = getattr(metagraph, "hotkeys", [])
        return (
            uid,
            await query_miner(
                client=client,
                uid=uid,
                hotkey=hotkeys[uid] if uid < len(hotkeys) else f"uid-{uid}",
                endpoint=f"http://{metagraph.axons[uid].ip}:{metagraph.axons[uid].port}",
                request=payload,
                wallet=runtime.wallet,
                timeout_seconds=runtime.sync_miner_request_timeout_seconds,
            ),
        )

    async with httpx.AsyncClient(
        timeout=runtime.sync_miner_request_timeout_seconds
    ) as client:
        tasks = [
            asyncio.create_task(run_query(uid))
            for uid in ranked_uids
        ]

        miner_results: dict[int, Any] = {}
        try:
            for completed in asyncio.as_completed(tasks):
                uid, result = await completed
                miner_results[uid] = result

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return [
        (completed_uid, miner_results[completed_uid])
        for completed_uid in ranked_uids
        if completed_uid in miner_results
    ]


async def _stream_ranked_miners(
    runtime: ValidatorGatewayRuntime,
    metagraph: bt.Metagraph,
    payload: VideoSearchRequest,
    ranked_uids: list[int],
):
    request_id = payload.request_id or "unknown-request"

    async def event_stream():
        yield _format_sse(
            "accepted",
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": payload.request_id,
                "status": "accepted",
                "queried_uids": ranked_uids,
            },
        )

        async with httpx.AsyncClient(
            timeout=runtime.stream_miner_request_timeout_seconds
        ) as client:
            async def run_query(uid: int):
                hotkeys = getattr(metagraph, "hotkeys", [])
                return (
                    uid,
                    await query_miner(
                        client=client,
                        uid=uid,
                        hotkey=hotkeys[uid] if uid < len(hotkeys) else f"uid-{uid}",
                        endpoint=f"http://{metagraph.axons[uid].ip}:{metagraph.axons[uid].port}",
                        request=payload,
                        wallet=runtime.wallet,
                        timeout_seconds=runtime.stream_miner_request_timeout_seconds,
                    ),
                )

            tasks = [asyncio.create_task(run_query(uid)) for uid in ranked_uids]
            miner_results: dict[int, Any] = {}
            try:
                for completed in asyncio.as_completed(tasks):
                    uid, result = await completed
                    miner_results[uid] = result

                    if result.response.results:
                        partial = _build_completed_response(
                            payload,
                            ranked_uids,
                            [
                                (completed_uid, miner_results[completed_uid])
                                for completed_uid in ranked_uids
                                if completed_uid in miner_results
                            ],
                        )
                        if partial is not None:
                            bt.logging.info(
                                f"[ORGANIC] Gateway request {request_id} stream update | uid={uid} | result_count={len(partial.results)}"
                            )
                            yield _format_sse(
                                "result",
                                {
                                    "protocol_version": PROTOCOL_VERSION,
                                    "request_id": partial.request_id,
                                    "status": "processing",
                                    "results": partial.model_dump(mode="json")["results"],
                                    "miner_metadata": partial.miner_metadata,
                                    "source_uid": uid,
                                },
                            )

                ordered_results = [
                    (completed_uid, miner_results[completed_uid])
                    for completed_uid in ranked_uids
                    if completed_uid in miner_results
                ]
                completed_response = _build_completed_response(
                    payload,
                    ranked_uids,
                    ordered_results,
                )

                if completed_response is not None:
                    bt.logging.success(
                        f"[ORGANIC] Gateway request {request_id} stream completed | successful_uids={completed_response.miner_metadata['selected_uids']} | returned_results={len(completed_response.results)}"
                    )
                    yield _format_sse(
                        "done",
                        completed_response.model_dump(mode="json"),
                    )
                    return

                failures = [
                    query_result.failure
                    for _, query_result in ordered_results
                    if query_result.failure
                ]
                error_code, error_message, status_code = _derive_gateway_failure(failures)
                bt.logging.warning(
                    f"[ORGANIC] Gateway request {request_id} stream failed | code={error_code} | queried_uids={ranked_uids}"
                )
                yield _format_sse(
                    "error",
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "request_id": payload.request_id,
                        "status": "failed",
                        "error": {
                            "code": error_code,
                            "message": error_message,
                            "details": {
                                "status_code": status_code,
                                "queried_uids": ranked_uids,
                            },
                        },
                    },
                )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def create_validator_gateway(runtime: ValidatorGatewayRuntime) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "status": "ok",
            "service": "validator-gateway",
            "protocol_versions": [PROTOCOL_VERSION],
        }

    @app.get("/capabilities")
    async def capabilities():
        return {
            "ok": True,
            "service": "validator-gateway",
            "protocol_versions": [PROTOCOL_VERSION],
        }

    async def handle_search(payload: VideoSearchRequest):
        request_id = payload.request_id or "unknown-request"
        bt.logging.info(
            f"[ORGANIC] Gateway request {request_id} | Query: {payload.query} | Video: {payload.video_url} | top_k={payload.top_k}"
        )
        metagraph = _get_metagraph_snapshot(runtime)
        ranked_uids = _rank_candidate_uids(runtime, metagraph)
        if not ranked_uids:
            bt.logging.warning(
                f"[ORGANIC] Gateway request {request_id} failed before dispatch: no responsive miners"
            )
            return build_protocol_error(
                code="INTERNAL_ERROR",
                message="No responsive miners are currently available.",
                status_code=503,
                request_id=payload.request_id,
            )

        bt.logging.info(
            f"[ORGANIC] Gateway request {request_id} | Candidate miners: {ranked_uids}"
        )

        miner_results = await _query_ranked_miners(
            runtime,
            metagraph,
            payload,
            ranked_uids,
        )

        aggregated_results = []
        successful_uids = []
        for uid, query_result in miner_results:
            if query_result.response.results:
                successful_uids.append(uid)
                aggregated_results.extend(query_result.response.results)

        if aggregated_results:
            ranked_results = _dedupe_and_rank_results(aggregated_results, payload.top_k)
            bt.logging.success(
                f"[ORGANIC] Gateway request {request_id} completed | Successful miners: {successful_uids} | Returned results: {len(ranked_results)}"
            )
            return VideoSearchResponse(
                request_id=payload.request_id,
                status="completed",
                results=ranked_results,
                miner_metadata={
                    "source": "validator-gateway",
                    "selected_uids": successful_uids,
                    "queried_uids": ranked_uids,
                },
            )

        failures = [
            query_result.failure
            for _, query_result in miner_results
            if query_result.failure
        ]
        if ranked_uids:
            error_code, error_message, status_code = _derive_gateway_failure(failures)
        else:
            error_code, error_message, status_code = (
                "INVALID_REQUEST",
                "No responsive miners are currently available.",
                503,
            )

        bt.logging.warning(
            f"[ORGANIC] Gateway request {request_id} failed | code={error_code} | queried_uids={ranked_uids}"
        )

        return build_protocol_error(
            code=error_code,
            message=error_message,
            status_code=status_code,
            request_id=payload.request_id,
            details={
                "queried_uids": ranked_uids,
                "miner_failures": [
                    {
                        "uid": uid,
                        "kind": query_result.failure.kind if query_result.failure else "empty_result",
                        "message": query_result.failure.message if query_result.failure else "Miner returned no results.",
                        "status_code": query_result.failure.status_code if query_result.failure else None,
                        "protocol_code": query_result.failure.protocol_code if query_result.failure else None,
                    }
                    for uid, query_result in miner_results
                ],
            },
        )

    @app.post("/search", response_model=VideoSearchResponse)
    async def search(payload: VideoSearchRequest):
        return await handle_search(payload)

    @app.post("/search/stream")
    async def search_stream(payload: VideoSearchRequest):
        request_id = payload.request_id or "unknown-request"
        bt.logging.info(
            f"[ORGANIC] Gateway stream request {request_id} | Query: {payload.query} | Video: {payload.video_url} | top_k={payload.top_k}"
        )
        metagraph = _get_metagraph_snapshot(runtime)
        ranked_uids = _rank_candidate_uids(runtime, metagraph)
        if not ranked_uids:
            return build_protocol_error(
                code="INTERNAL_ERROR",
                message="No responsive miners are currently available.",
                status_code=503,
                request_id=payload.request_id,
            )

        bt.logging.info(
            f"[ORGANIC] Gateway stream request {request_id} | Candidate miners: {ranked_uids}"
        )
        return await _stream_ranked_miners(runtime, metagraph, payload, ranked_uids)

    return app
