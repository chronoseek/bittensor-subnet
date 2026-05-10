import asyncio
import os
from dataclasses import dataclass

import bittensor as bt
import httpx

from chronoseek.chain.submissions import MinerSubmission


@dataclass(frozen=True)
class ChutesRuntimeEndpoint:
    uid: int
    hotkey: str
    endpoint: str
    submission: MinerSubmission | None = None


def chutes_auth_headers_from_env(
    *,
    require_token: bool = False,
    include_content_type: bool = False,
) -> dict[str, str]:
    chutes_api_key = os.getenv("CHUTES_API_KEY")
    if require_token and not chutes_api_key:
        raise RuntimeError("CHUTES_API_KEY is required for Chutes API calls.")

    headers: dict[str, str] = {}

    from chutes._version import version as current_version
    headers["X-Chutes-Version"] = current_version

    if chutes_api_key:
        # headers["Authorization"] = f"Bearer {chutes_api_key}"
        headers["Authorization"] = chutes_api_key
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def resolve_submission_endpoint(
    submission: MinerSubmission,
    *,
    chutes_base_domain: str,
) -> str | None:
    if submission.endpoint:
        return str(submission.endpoint).rstrip("/")

    if submission.chute_slug:
        domain = chutes_base_domain.strip().removeprefix("https://").removeprefix(
            "http://"
        )
        return f"https://{submission.chute_slug}.{domain}".rstrip("/")

    return None


def build_evaluation_endpoints(
    *,
    metagraph: bt.Metagraph,
    candidate_uids: list[int] | None,
    submissions_by_hotkey: dict[str, MinerSubmission] | None = None,
    chutes_base_domain: str = "chutes.ai",
) -> list[ChutesRuntimeEndpoint]:
    submissions_by_hotkey = submissions_by_hotkey or {}
    uids_to_query = candidate_uids if candidate_uids is not None else metagraph.uids
    endpoints: list[ChutesRuntimeEndpoint] = []
    hotkeys = getattr(metagraph, "hotkeys", [])

    for raw_uid in uids_to_query:
        uid = int(raw_uid)
        if uid < 0 or uid >= len(hotkeys):
            continue

        hotkey = hotkeys[uid]
        submission = submissions_by_hotkey.get(hotkey)
        if submission is None:
            continue

        endpoint = resolve_submission_endpoint(
            submission,
            chutes_base_domain=chutes_base_domain,
        )
        if endpoint:
            endpoints.append(
                ChutesRuntimeEndpoint(
                    uid=uid,
                    hotkey=hotkey,
                    endpoint=endpoint,
                    submission=submission,
                )
            )

    return endpoints


def build_runtime_endpoints_from_map(
    *,
    metagraph: bt.Metagraph,
    endpoint_map: dict[int, str],
    candidate_uids: list[int],
) -> list[ChutesRuntimeEndpoint]:
    endpoints: list[ChutesRuntimeEndpoint] = []
    hotkeys = getattr(metagraph, "hotkeys", [])
    for raw_uid in candidate_uids:
        uid = int(raw_uid)
        if uid < 0 or uid >= len(hotkeys):
            continue
        endpoint = endpoint_map.get(uid)
        if endpoint:
            endpoints.append(
                ChutesRuntimeEndpoint(
                    uid=uid,
                    hotkey=hotkeys[uid],
                    endpoint=endpoint,
                )
            )
    return endpoints


def build_submission_endpoint_map(
    *,
    metagraph: bt.Metagraph,
    submissions_by_hotkey: dict[str, MinerSubmission],
    chutes_base_domain: str = "chutes.ai",
) -> dict[int, str]:
    """Resolve v2 miner submissions for hotkeys present in the metagraph."""

    endpoints: dict[int, str] = {}
    hotkeys = getattr(metagraph, "hotkeys", [])
    for uid, hotkey in enumerate(hotkeys):
        submission = submissions_by_hotkey.get(hotkey)
        if submission is None:
            continue
        endpoint = resolve_submission_endpoint(
            submission,
            chutes_base_domain=chutes_base_domain,
        )
        if endpoint:
            endpoints[int(uid)] = endpoint
    return endpoints


async def check_runtime_health(
    *,
    client: httpx.AsyncClient,
    uid: int,
    endpoint: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> bool:
    subject = f"Miner UID {uid}" if int(uid) >= 0 else "Runtime"
    try:
        response = await client.get(
            f"{endpoint.rstrip('/')}/health",
            headers=headers or {},
            timeout=max(0.5, float(timeout_seconds)),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            bt.logging.debug(
                f"{subject} health check returned unexpected payload: {payload}"
            )
            return False
        return True
    except Exception as exc:
        bt.logging.debug(
            f"{subject} failed /health at {endpoint}: {exc}"
        )
        return False


async def filter_healthy_runtime_endpoints(
    *,
    endpoint_map: dict[int, str],
    health_timeout_seconds: float,
    provider_headers: dict[str, str] | None = None,
) -> dict[int, str]:
    healthy_endpoint_map: dict[int, str] = {}
    if not endpoint_map:
        return healthy_endpoint_map

    async with httpx.AsyncClient(timeout=max(0.5, float(health_timeout_seconds))) as client:
        checks = {
            uid: check_runtime_health(
                client=client,
                uid=int(uid),
                endpoint=endpoint,
                timeout_seconds=health_timeout_seconds,
                headers=provider_headers,
            )
            for uid, endpoint in endpoint_map.items()
        }
        for uid, is_healthy in zip(checks, await asyncio.gather(*checks.values())):
            if is_healthy:
                healthy_endpoint_map[int(uid)] = endpoint_map[int(uid)]

    return healthy_endpoint_map
