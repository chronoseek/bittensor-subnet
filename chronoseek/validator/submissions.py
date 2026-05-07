import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import bittensor as bt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)


CHRONOSEEK_RUNTIME_PROTOCOL = "chronoseek-runtime-v2"


class MinerSubmission(BaseModel):
    """Structured v2 miner submission metadata.

    The chain payload is intentionally transport/runtime metadata only. Auth
    secrets stay local to validators as environment variables.
    """

    model_config = ConfigDict(extra="allow")

    version: str = "2.0"
    runtime: Literal["chutes"] = "chutes"
    protocol: str = CHRONOSEEK_RUNTIME_PROTOCOL
    hotkey: str | None = None
    uid: int | None = Field(default=None, ge=0)
    endpoint: HttpUrl | None = None
    chute_id: str | None = None
    chute_slug: str | None = None
    artifact_id: str | None = None
    artifact_revision: str | None = None
    artifact_digest: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    created_at_block: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_aliases(cls, data: Any):
        if not isinstance(data, dict):
            return data

        migrated = dict(data)
        if "endpoint" not in migrated:
            for key in ("base_url", "url"):
                if key in migrated:
                    migrated["endpoint"] = migrated[key]
                    break
        if "chute_slug" not in migrated and "slug" in migrated:
            migrated["chute_slug"] = migrated["slug"]
        if "artifact_id" not in migrated and "model" in migrated:
            migrated["artifact_id"] = migrated["model"]
        if "artifact_revision" not in migrated and "revision" in migrated:
            migrated["artifact_revision"] = migrated["revision"]
        return migrated

    @model_validator(mode="after")
    def validate_submission(self):
        if self.protocol != CHRONOSEEK_RUNTIME_PROTOCOL:
            raise ValueError(
                f"unsupported submission protocol {self.protocol!r}; expected {CHRONOSEEK_RUNTIME_PROTOCOL!r}"
            )
        if not any([self.endpoint, self.chute_slug, self.chute_id]):
            raise ValueError("submission must include endpoint, chute_slug, or chute_id")
        return self


@dataclass(frozen=True)
class MinerEvaluationEndpoint:
    uid: int
    hotkey: str
    endpoint: str
    submission: MinerSubmission | None = None


def _patch_bittensor_commit_decoder() -> None:
    """Patch Bittensor 10.x revealed commitment decoding when needed.

    Affine carries the same workaround. Some Bittensor 10.x versions decode
    revealed string commitments as hex and fail when the chain returns raw
    SCALE-prefixed bytes represented as a Python string.
    """

    try:
        from bittensor.core.chain_data import utils as bt_utils
        from bittensor.core import async_subtensor as bt_async
    except Exception as exc:
        bt.logging.debug(f"Skipping commitment decoder patch: {exc}")
        return

    if getattr(bt_utils, "_chronoseek_safe_decode_patched", False):
        return

    def scale_offset(first_byte: int) -> int:
        mode = first_byte & 0b11
        if mode == 0:
            return 1
        if mode == 1:
            return 2
        return 4

    def to_bytes(value) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            stripped = value.removeprefix("0x")
            try:
                return bytes.fromhex(stripped)
            except ValueError:
                return value.encode("latin-1", errors="replace")
        return bytes(value)

    def safe_decode(encoded_data):
        commitment, revealed_block = encoded_data
        raw = to_bytes(commitment)
        offset = scale_offset(raw[0]) if raw else 0
        return revealed_block, raw[offset:].decode("utf-8", errors="ignore")

    def safe_decode_with_hotkey(encoded_data):
        hotkey, data = encoded_data
        decoded = []
        for item in data:
            try:
                decoded.append(safe_decode(item))
            except Exception as exc:
                bt.logging.warning(
                    f"Skipping malformed revealed commitment for {hotkey}: {exc}"
                )
        return hotkey, tuple(decoded)

    bt_utils.decode_revealed_commitment = safe_decode
    bt_utils.decode_revealed_commitment_with_hotkey = safe_decode_with_hotkey
    bt_async.decode_revealed_commitment_with_hotkey = safe_decode_with_hotkey
    bt_utils._chronoseek_safe_decode_patched = True


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _coerce_submission(
    *,
    raw: Any,
    hotkey: str | None = None,
    uid: int | None = None,
    created_at_block: int | None = None,
) -> MinerSubmission | None:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return None

    payload = dict(raw)
    if hotkey is not None:
        payload.setdefault("hotkey", hotkey)
    if uid is not None:
        payload.setdefault("uid", uid)
    if created_at_block is not None:
        payload.setdefault("created_at_block", created_at_block)

    try:
        return MinerSubmission(**payload)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        bt.logging.debug(f"Rejected miner submission payload: {exc}")
        return None


async def load_chain_submissions(
    subtensor: Any,
    netuid: int,
    metagraph: bt.Metagraph,
) -> dict[str, MinerSubmission]:
    """Read latest revealed v2 submissions from the chain by hotkey."""

    if not hasattr(subtensor, "get_all_revealed_commitments"):
        bt.logging.warning(
            "Configured chain submission routing, but this subtensor does not provide get_all_revealed_commitments."
        )
        return {}

    _patch_bittensor_commit_decoder()
    try:
        commits = await _maybe_await(subtensor.get_all_revealed_commitments(netuid))
    except Exception as exc:
        bt.logging.warning(f"Failed to read miner submissions from chain: {exc}")
        return {}

    submissions: dict[str, MinerSubmission] = {}
    hotkeys = getattr(metagraph, "hotkeys", [])
    for uid, hotkey in enumerate(hotkeys):
        hotkey_commits = commits.get(hotkey) if isinstance(commits, dict) else None
        if not hotkey_commits:
            continue
        try:
            block, commit_data = hotkey_commits[-1]
            submission = _coerce_submission(
                raw=commit_data,
                hotkey=hotkey,
                uid=uid,
                created_at_block=int(block),
            )
        except Exception as exc:
            bt.logging.debug(
                f"Failed to parse latest submission for uid={uid} hotkey={hotkey}: {exc}"
            )
            continue
        if submission:
            submissions[hotkey] = submission

    return submissions


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
) -> list[MinerEvaluationEndpoint]:
    submissions_by_hotkey = submissions_by_hotkey or {}
    uids_to_query = candidate_uids if candidate_uids is not None else metagraph.uids
    endpoints: list[MinerEvaluationEndpoint] = []
    hotkeys = getattr(metagraph, "hotkeys", [])

    for raw_uid in uids_to_query:
        uid = int(raw_uid)
        if uid < 0 or uid >= len(hotkeys):
            continue

        hotkey = hotkeys[uid]
        submission = submissions_by_hotkey.get(hotkey)

        if submission is not None:
            endpoint = resolve_submission_endpoint(
                submission,
                chutes_base_domain=chutes_base_domain,
            )
            if endpoint:
                endpoints.append(
                    MinerEvaluationEndpoint(
                        uid=uid,
                        hotkey=hotkey,
                        endpoint=endpoint,
                        submission=submission,
                    )
                )

    return endpoints


def build_submission_endpoint_map(
    *,
    metagraph: bt.Metagraph,
    submissions_by_hotkey: dict[str, MinerSubmission],
    chutes_base_domain: str = "chutes.ai",
) -> dict[int, str]:
    """Resolve v2 submission endpoints for registered metagraph hotkeys."""

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


class MinerSubmissionResolver:
    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 300.0,
    ):
        self.cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._cached_at = 0.0
        self._cached: dict[str, MinerSubmission] = {}

    async def get_submissions(
        self,
        *,
        subtensor: Any,
        netuid: int,
        metagraph: bt.Metagraph,
    ) -> dict[str, MinerSubmission]:
        now = time.time()
        if self._cached_at and (now - self._cached_at) < self.cache_ttl_seconds:
            return dict(self._cached)

        submissions = await load_chain_submissions(subtensor, netuid, metagraph)

        self._cached = dict(submissions)
        self._cached_at = now
        bt.logging.info(
            f"Loaded {len(submissions)} v2 miner submissions from chain."
        )
        return submissions


def chutes_auth_headers_from_env() -> dict[str, str]:
    token = os.getenv("CHRONOSEEK_CHUTES_API_KEY") or os.getenv("CHUTES_API_KEY")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}
