from dataclasses import dataclass
from threading import Lock
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import bittensor as bt

from chronoseek.protocol_models import ProtocolError, VideoSearchRequest, VideoSearchResponse
from chronoseek.validator.forward import query_miner


@dataclass
class ValidatorGatewayRuntime:
    wallet: bt.Wallet
    metagraph: bt.Metagraph
    scores: np.ndarray
    score_lock: Lock
    max_miners_per_request: int


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


def _rank_candidate_uids(runtime: ValidatorGatewayRuntime) -> list[int]:
    with runtime.score_lock:
        current_scores = np.array(runtime.scores, copy=True)

    candidates: list[tuple[int, float]] = []
    for uid in runtime.metagraph.uids:
        axon = runtime.metagraph.axons[uid]
        if axon.ip == "0.0.0.0":
            continue
        score = float(current_scores[int(uid)]) if int(uid) < len(current_scores) else 0.0
        candidates.append((int(uid), score))

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


def create_validator_gateway(runtime: ValidatorGatewayRuntime) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/search", response_model=VideoSearchResponse)
    async def search(payload: VideoSearchRequest):
        ranked_uids = _rank_candidate_uids(runtime)
        if not ranked_uids:
            return build_protocol_error(
                code="INTERNAL_ERROR",
                message="No reachable miners are currently available.",
                status_code=503,
                request_id=payload.request_id,
            )

        async with httpx.AsyncClient(timeout=30.0) as client:
            results = []
            for uid in ranked_uids:
                axon = runtime.metagraph.axons[uid]
                endpoint = f"http://{axon.ip}:{axon.port}"
                query_result = await query_miner(
                    client=client,
                    endpoint=endpoint,
                    request=payload,
                    wallet=runtime.wallet,
                )
                results.append((uid, query_result))

        aggregated_results = []
        successful_uids = []
        for uid, query_result in results:
            if query_result.response.results:
                successful_uids.append(uid)
                aggregated_results.extend(query_result.response.results)

        if aggregated_results:
            return VideoSearchResponse(
                request_id=payload.request_id,
                status="completed",
                results=_dedupe_and_rank_results(aggregated_results, payload.top_k),
                miner_metadata={
                    "source": "validator-gateway",
                    "selected_uids": successful_uids,
                    "queried_uids": ranked_uids,
                },
            )

        failures = [query_result.failure for _, query_result in results if query_result.failure]
        if failures and all(failure.kind == "timeout" for failure in failures):
            error_code = "TIMEOUT"
            error_message = "All miner queries timed out."
            status_code = 504
        elif ranked_uids:
            error_code = "INTERNAL_ERROR"
            error_message = "No miner returned a usable search result."
            status_code = 502
        else:
            error_code = "INVALID_REQUEST"
            error_message = "No reachable miners are currently available."
            status_code = 503

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
                    for uid, query_result in results
                ],
            },
        )

    return app
