import inspect
import json
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
PERMANENT_SUBMISSION_ERROR = (
    "This hotkey has already submitted a ChronoSeek miner commitment. "
    "Miner submissions are permanent and cannot be edited or replaced. "
    "If this miner was disqualified or needs a different runtime, register a new hotkey and submit again."
)


class DuplicateMinerSubmissionError(RuntimeError):
    """Raised when a hotkey tries to submit more than one permanent commitment."""


@dataclass(frozen=True)
class ChainSubmissionSnapshot:
    submissions: dict[str, "MinerSubmission"]
    duplicate_hotkeys: set[str]


class MinerSubmission(BaseModel):
    """Structured v2 miner runtime metadata committed on-chain.

    The chain payload is transport/runtime metadata only. Runtime auth tokens
    and provider credentials stay local to validators and deployment tools.
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


def serialize_submission(submission: MinerSubmission) -> str:
    payload = submission.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def commit_miner_submission(
    *,
    subtensor: Any,
    wallet: bt.Wallet,
    netuid: int,
    submission: MinerSubmission,
    blocks_until_reveal: int,
) -> bool:
    wallet_hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if wallet_hotkey and await hotkey_has_revealed_commitment(
        subtensor=subtensor,
        netuid=netuid,
        hotkey=wallet_hotkey,
    ):
        raise DuplicateMinerSubmissionError(PERMANENT_SUBMISSION_ERROR)

    result = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=int(netuid),
        data=serialize_submission(submission),
        blocks_until_reveal=max(1, int(blocks_until_reveal)),
    )
    response = await maybe_await(result)
    return bool(getattr(response, "success", response))


def _sorted_hotkey_commits(hotkey_commits: Any) -> list[tuple[int, Any]]:
    if not hotkey_commits:
        return []

    commits = []
    for item in hotkey_commits:
        try:
            block, commit_data = item
            commits.append((int(block), commit_data))
        except (TypeError, ValueError):
            continue
    return sorted(commits, key=lambda pair: pair[0])


async def get_hotkey_revealed_commitments(
    *,
    subtensor: Any,
    netuid: int,
    hotkey: str,
) -> list[tuple[int, Any]]:
    if not hasattr(subtensor, "get_all_revealed_commitments"):
        return []

    _patch_bittensor_commit_decoder()
    try:
        commits = await maybe_await(subtensor.get_all_revealed_commitments(int(netuid)))
    except Exception as exc:
        bt.logging.warning(f"Failed to read miner submissions from chain: {exc}")
        return []

    if not isinstance(commits, dict):
        return []
    return _sorted_hotkey_commits(commits.get(hotkey))


async def hotkey_has_revealed_commitment(
    *,
    subtensor: Any,
    netuid: int,
    hotkey: str,
) -> bool:
    return bool(
        await get_hotkey_revealed_commitments(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
        )
    )


def _patch_bittensor_commit_decoder() -> None:
    """Patch Bittensor 10.x revealed commitment decoding when needed."""

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
    snapshot = await load_chain_submission_snapshot(
        subtensor=subtensor,
        netuid=netuid,
        metagraph=metagraph,
    )
    return snapshot.submissions


async def load_chain_submission_snapshot(
    subtensor: Any,
    netuid: int,
    metagraph: bt.Metagraph,
) -> ChainSubmissionSnapshot:
    """Read valid v2 miner submissions and duplicate-submit hotkeys from chain.

    Miner submissions are permanent. A hotkey with more than one revealed
    commitment is disqualified instead of being allowed to keep its first
    submission, because multiple submissions violate the one-runtime-per-hotkey
    rule.
    """

    if not hasattr(subtensor, "get_all_revealed_commitments"):
        bt.logging.warning(
            "Chain submission routing is configured, but this subtensor does not provide get_all_revealed_commitments."
        )
        return ChainSubmissionSnapshot(submissions={}, duplicate_hotkeys=set())

    _patch_bittensor_commit_decoder()
    try:
        commits = await maybe_await(subtensor.get_all_revealed_commitments(netuid))
    except Exception as exc:
        bt.logging.warning(f"Failed to read miner submissions from chain: {exc}")
        return ChainSubmissionSnapshot(submissions={}, duplicate_hotkeys=set())

    submissions: dict[str, MinerSubmission] = {}
    duplicate_hotkeys: set[str] = set()
    hotkeys = getattr(metagraph, "hotkeys", [])
    for uid, hotkey in enumerate(hotkeys):
        hotkey_commits = commits.get(hotkey) if isinstance(commits, dict) else None
        if not hotkey_commits:
            continue
        try:
            sorted_commits = _sorted_hotkey_commits(hotkey_commits)
            if not sorted_commits:
                continue
            if len(sorted_commits) > 1:
                duplicate_hotkeys.add(str(hotkey))
                bt.logging.warning(
                    f"Disqualifying hotkey {hotkey} with {len(sorted_commits)} revealed miner submissions; miner submissions are permanent and multiple submissions score zero."
                )
                continue
            block, commit_data = sorted_commits[0]
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

    return ChainSubmissionSnapshot(
        submissions=submissions,
        duplicate_hotkeys=duplicate_hotkeys,
    )


class MinerSubmissionResolver:
    def __init__(
        self,
        *,
        cache_ttl_seconds: float = 300.0,
    ):
        self.cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._cached_at = 0.0
        self._cached: dict[str, MinerSubmission] = {}
        self._cached_duplicate_hotkeys: set[str] = set()

    def get_duplicate_hotkeys(self) -> set[str]:
        return set(self._cached_duplicate_hotkeys)

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

        snapshot = await load_chain_submission_snapshot(
            subtensor=subtensor,
            netuid=netuid,
            metagraph=metagraph,
        )
        submissions = snapshot.submissions
        self._cached = dict(submissions)
        self._cached_duplicate_hotkeys = set(snapshot.duplicate_hotkeys)
        self._cached_at = now
        bt.logging.info(
            "Loaded v2 miner submissions from chain | "
            f"valid={len(submissions)} | duplicate_hotkeys={len(snapshot.duplicate_hotkeys)}"
        )
        return submissions
