"""Shared, small standalone helpers that don't belong to one specific module."""

from typing import Any

from chronoseek.chain.submissions import MinerSubmission
from chronoseek.chutes.runtime import normalize_endpoint_scheme, resolve_submission_endpoint
from chronoseek.constants import DEFAULT_CHUTES_BASE_DOMAIN


def resolve_axon_endpoint(neuron: Any) -> str | None:
    """Return a fully-qualified base URL for a neuron's served axon, or None.

    Bittensor 11's `MetagraphNeuron.axon` is already the served `ip:port`
    string, or None when nothing is served (no separate AxonInfo object).
    Normalized here via `normalize_endpoint_scheme` as defense in depth, but
    `check_runtime_health` (chronoseek/chutes/runtime.py) also normalizes
    whatever endpoint it's given directly, so this doesn't depend on every
    caller resolving endpoints through this specific function.
    """
    if neuron is None:
        return None
    axon = getattr(neuron, "axon", None)
    if not axon:
        return None
    axon = str(axon).strip()
    if not axon:
        return None
    return normalize_endpoint_scheme(axon)


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
