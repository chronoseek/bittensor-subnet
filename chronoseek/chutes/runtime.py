import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

from chronoseek.chain.submissions import MinerSubmission
from chronoseek.constants import DEFAULT_CHUTES_BASE_DOMAIN
from chronoseek.logging import logger


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
        headers["Authorization"] = f"Bearer {chutes_api_key}"
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
    metagraph: Any,
    candidate_uids: list[int] | None,
    submissions_by_hotkey: dict[str, MinerSubmission] | None = None,
    chutes_base_domain: str = DEFAULT_CHUTES_BASE_DOMAIN,
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
    metagraph: Any,
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
    metagraph: Any,
    submissions_by_hotkey: dict[str, MinerSubmission],
    chutes_base_domain: str = DEFAULT_CHUTES_BASE_DOMAIN,
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


def resolve_axon_endpoint(neuron: Any) -> str | None:
    """Return a fully-qualified base URL for a neuron's served axon, or None.

    Bittensor 11's `MetagraphNeuron.axon` is already the served `ip:port`
    string, or None when nothing is served (no separate AxonInfo object).
    A scheme is required here: unlike forward.py's query path (which
    normalizes bare `ip:port` via `_normalize_endpoint`), this module's own
    `check_runtime_health()` builds the health-check URL directly and does
    not normalize it, so an unqualified `ip:port` fails outright with
    httpx.UnsupportedProtocol.
    """
    if neuron is None:
        return None
    axon = getattr(neuron, "axon", None)
    if not axon:
        return None
    axon = str(axon).strip()
    if not axon:
        return None
    return axon if axon.startswith(("http://", "https://")) else f"http://{axon}"


def build_endpoint_map_with_axon_fallback(
    *,
    metagraph: Any,
    submissions_by_hotkey: dict[str, MinerSubmission],
    chutes_base_domain: str = DEFAULT_CHUTES_BASE_DOMAIN,
    disqualified_uids: set[int] | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Resolve each miner's per-refresh-cycle endpoint.

    Committed Chutes metadata always wins when it resolves to an endpoint.
    The miner's axon is only used as a fallback when there is no resolvable
    Chutes endpoint at all (missing commitment, or a chute_id-only
    commitment that doesn't resolve to a URL). There is no same-round
    crossover: exactly one source is chosen per uid.

    Returns (endpoint_map, endpoint_sources), where endpoint_sources values
    are "chutes" or "axon".
    """
    disqualified_uids = disqualified_uids or set()
    endpoint_map: dict[int, str] = {}
    endpoint_sources: dict[int, str] = {}
    hotkeys = getattr(metagraph, "hotkeys", [])
    neurons = getattr(metagraph, "neurons", None) or []

    for uid, hotkey in enumerate(hotkeys):
        submission = submissions_by_hotkey.get(hotkey)
        chutes_endpoint = (
            resolve_submission_endpoint(submission, chutes_base_domain=chutes_base_domain)
            if submission is not None
            else None
        )
        if chutes_endpoint:
            endpoint_map[int(uid)] = chutes_endpoint
            endpoint_sources[int(uid)] = "chutes"
            continue

        if int(uid) in disqualified_uids:
            continue

        neuron = neurons[uid] if uid < len(neurons) else None
        axon_endpoint = resolve_axon_endpoint(neuron)
        if axon_endpoint:
            endpoint_map[int(uid)] = axon_endpoint
            endpoint_sources[int(uid)] = "axon"

    return endpoint_map, endpoint_sources


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
            logger.debug(
                f"{subject} health check returned unexpected payload: {payload}"
            )
            return False
        return True
    except Exception as exc:
        logger.debug(
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
