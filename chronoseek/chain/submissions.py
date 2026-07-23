import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

import bittensor as bt
from bittensor_core import get_encrypted_commitment
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from chronoseek.logging import logger


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
    enforce_one_hotkey_one_submission: bool = True,
) -> bool:
    wallet_hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if (
        enforce_one_hotkey_one_submission
        and wallet_hotkey
        and await hotkey_has_revealed_commitment(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=wallet_hotkey,
        )
    ):
        raise DuplicateMinerSubmissionError(PERMANENT_SUBMISSION_ERROR)

    encrypted, reveal_round = get_encrypted_commitment(
        serialize_submission(submission),
        max(1, int(blocks_until_reveal)),
        12,
    )
    call = bt.calls.Commitments.set_commitment(
        netuid=int(netuid),
        info={
            "fields": [
                [
                    {
                        "TimelockEncrypted": {
                            "encrypted": encrypted,
                            "reveal_round": reveal_round,
                        }
                    }
                ]
            ]
        },
    )
    result = subtensor.submit_call(
        call,
        wallet,
        signer="hotkey",
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    response = await maybe_await(result)
    return bool(response)


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
    subnet_api = getattr(subtensor, "subnets", None)
    if subnet_api is None or not hasattr(subnet_api, "commitments"):
        return []

    try:
        commits = await maybe_await(subnet_api.commitments(int(netuid)))
    except Exception as exc:
        logger.warning(f"Failed to read miner submissions from chain: {exc}")
        return []

    if not isinstance(commits, dict):
        return []
    return _revealed_commitment_history(commits.get(hotkey))


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


def _revealed_commitment_history(commitment: Any) -> list[tuple[int, Any]]:
    """Return all readable values from a Bittensor 11 commitment record."""

    if commitment is None:
        return []

    history = _sorted_hotkey_commits(getattr(commitment, "revealed", []))
    encrypted = bool(getattr(commitment, "encrypted", False))
    value = getattr(commitment, "value", None)
    if not encrypted and value is not None:
        current = (int(getattr(commitment, "block", 0)), value)
        if current not in history:
            history.append(current)
    return sorted(history, key=lambda pair: pair[0])


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
        logger.debug(f"Rejected miner submission payload: {exc}")
        return None


async def load_chain_submissions(
    subtensor: Any,
    netuid: int,
    metagraph: Any,
    *,
    enforce_one_hotkey_one_submission: bool = True,
) -> dict[str, MinerSubmission]:
    snapshot = await load_chain_submission_snapshot(
        subtensor=subtensor,
        netuid=netuid,
        metagraph=metagraph,
        enforce_one_hotkey_one_submission=enforce_one_hotkey_one_submission,
    )
    return snapshot.submissions


async def load_chain_submission_snapshot(
    subtensor: Any,
    netuid: int,
    metagraph: Any,
    *,
    enforce_one_hotkey_one_submission: bool = True,
) -> ChainSubmissionSnapshot:
    """Read valid v2 miner submissions and duplicate-submit hotkeys from chain.

    By default, a hotkey with more than one revealed commitment is
    disqualified. When enforcement is disabled, the newest revealed submission
    is used and no duplicate penalty is reported.
    """

    subnet_api = getattr(subtensor, "subnets", None)
    if subnet_api is None or not hasattr(subnet_api, "commitments"):
        logger.warning(
            "Chain submission routing is configured, but this subtensor does not provide subnets.commitments."
        )
        return ChainSubmissionSnapshot(submissions={}, duplicate_hotkeys=set())

    try:
        commits = await maybe_await(subnet_api.commitments(int(netuid)))
    except Exception as exc:
        logger.warning(f"Failed to read miner submissions from chain: {exc}")
        return ChainSubmissionSnapshot(submissions={}, duplicate_hotkeys=set())

    submissions: dict[str, MinerSubmission] = {}
    duplicate_hotkeys: set[str] = set()
    hotkeys = getattr(metagraph, "hotkeys", [])
    for uid, hotkey in enumerate(hotkeys):
        commitment = commits.get(hotkey) if isinstance(commits, dict) else None
        if commitment is None:
            continue
        try:
            sorted_commits = _revealed_commitment_history(commitment)
            if not sorted_commits:
                continue
            if len(sorted_commits) > 1 and enforce_one_hotkey_one_submission:
                duplicate_hotkeys.add(str(hotkey))
                logger.warning(
                    f"Disqualifying hotkey {hotkey} with {len(sorted_commits)} revealed miner submissions; miner submissions are permanent and multiple submissions score zero."
                )
                continue
            block, commit_data = sorted_commits[-1]
            submission = _coerce_submission(
                raw=commit_data,
                hotkey=hotkey,
                uid=uid,
                created_at_block=int(block),
            )
        except Exception as exc:
            logger.debug(
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
        enforce_one_hotkey_one_submission: bool = True,
    ):
        self.cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self.enforce_one_hotkey_one_submission = bool(
            enforce_one_hotkey_one_submission
        )
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
        metagraph: Any,
    ) -> dict[str, MinerSubmission]:
        now = time.time()
        if self._cached_at and (now - self._cached_at) < self.cache_ttl_seconds:
            return dict(self._cached)

        snapshot = await load_chain_submission_snapshot(
            subtensor=subtensor,
            netuid=netuid,
            metagraph=metagraph,
            enforce_one_hotkey_one_submission=self.enforce_one_hotkey_one_submission,
        )
        submissions = snapshot.submissions
        self._cached = dict(submissions)
        self._cached_duplicate_hotkeys = set(snapshot.duplicate_hotkeys)
        self._cached_at = now
        logger.info(
            "Loaded v2 miner submissions from chain | "
            f"valid={len(submissions)} | duplicate_hotkeys={len(snapshot.duplicate_hotkeys)}"
        )
        return submissions
