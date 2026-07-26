"""Live chain commit/fetch checks."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from chronoseek.chain.submissions import (
    DuplicateMinerSubmissionError,
    MinerSubmission,
    PERMANENT_SUBMISSION_ERROR,
    commit_miner_submission,
    get_hotkey_revealed_commitments,
    load_chain_submission_snapshot,
)
from chronoseek.bittensor_sdk import fetch_metagraph
from chronoseek.constants import DEFAULT_NETUID, DEFAULT_NETWORK
from live_helpers import env_value, require_extra_confirmation, require_live


pytestmark = pytest.mark.live


def live_subtensor_metagraph():
    import bittensor as bt

    subtensor = bt.Subtensor(network=env_value("NETWORK", DEFAULT_NETWORK))
    metagraph = fetch_metagraph(
        subtensor,
        int(env_value("NETUID", str(DEFAULT_NETUID))),
    )
    return subtensor, metagraph


def live_bittensor_objects():
    import bittensor as bt

    wallet = bt.Wallet(
        name=env_value("WALLET_NAME", "default"),
        hotkey=env_value("HOTKEY_NAME", "default"),
        path=str(Path(env_value("WALLET_PATH", "~/.bittensor/wallets/")).expanduser()),
    )
    subtensor, metagraph = live_subtensor_metagraph()
    return wallet, subtensor, metagraph


def test_live_chain_commit_submission() -> None:
    """Commit one live miner runtime submission to chain."""

    require_live("CHRONOSEEK_LIVE_CHAIN", "WALLET_NAME", "HOTKEY_NAME", "NETWORK", "NETUID")
    require_extra_confirmation("CHRONOSEEK_LIVE_CHAIN_COMMIT", "chain commit")

    wallet, subtensor, metagraph = live_bittensor_objects()
    hotkey = wallet.hotkey.ss58_address
    assert hotkey in metagraph.hotkeys
    existing_commitments = asyncio.run(
        get_hotkey_revealed_commitments(
            subtensor=subtensor,
            netuid=int(env_value("NETUID", str(DEFAULT_NETUID))),
            hotkey=hotkey,
        )
    )
    if existing_commitments:
        pytest.skip(
            "this hotkey already has a revealed miner submission; use a fresh hotkey "
            "for commit testing or run the duplicate-submission rejection test"
        )

    submission = MinerSubmission(
        hotkey=hotkey,
        chute_slug=env_value(
            "LIVE_CHAIN_CHUTE_SLUG",
            f"alice-chronoseek-runtime-live-{uuid.uuid4().hex[:12]}",
        ),
        artifact_revision=env_value("LIVE_CHAIN_ARTIFACT_REVISION", uuid.uuid4().hex),
    )
    success = asyncio.run(
        commit_miner_submission(
            subtensor=subtensor,
            wallet=wallet,
            netuid=int(env_value("NETUID", str(DEFAULT_NETUID))),
            submission=submission,
            blocks_until_reveal=int(env_value("LIVE_CHAIN_BLOCKS_UNTIL_REVEAL", "1")),
        )
    )
    assert success is True


def test_live_chain_rejects_duplicate_submission_for_hotkey() -> None:
    """Verify a hotkey with an existing revealed submission cannot submit again."""

    require_live("CHRONOSEEK_LIVE_CHAIN", "WALLET_NAME", "HOTKEY_NAME", "NETWORK", "NETUID")
    require_extra_confirmation(
        "CHRONOSEEK_LIVE_CHAIN_DUPLICATE_COMMIT",
        "duplicate chain commit rejection",
    )

    wallet, subtensor, metagraph = live_bittensor_objects()
    hotkey = wallet.hotkey.ss58_address
    netuid = int(env_value("NETUID", str(DEFAULT_NETUID)))
    assert hotkey in metagraph.hotkeys

    existing_commitments = asyncio.run(
        get_hotkey_revealed_commitments(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
        )
    )
    if not existing_commitments:
        pytest.skip(
            "duplicate rejection requires a hotkey with an existing revealed "
            "submission; run the live commit test with this hotkey first and wait "
            "for reveal, or use an already-submitted hotkey"
        )

    submission = MinerSubmission(
        hotkey=hotkey,
        chute_slug=env_value(
            "LIVE_CHAIN_DUPLICATE_CHUTE_SLUG",
            f"alice-chronoseek-runtime-duplicate-{uuid.uuid4().hex[:12]}",
        ),
        artifact_revision=env_value(
            "LIVE_CHAIN_DUPLICATE_ARTIFACT_REVISION",
            uuid.uuid4().hex,
        ),
    )

    with pytest.raises(DuplicateMinerSubmissionError) as exc:
        asyncio.run(
            commit_miner_submission(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                submission=submission,
                blocks_until_reveal=int(
                    env_value("LIVE_CHAIN_BLOCKS_UNTIL_REVEAL", "1")
                ),
            )
        )

    assert str(exc.value) == PERMANENT_SUBMISSION_ERROR
    after_commitments = asyncio.run(
        get_hotkey_revealed_commitments(
            subtensor=subtensor,
            netuid=netuid,
            hotkey=hotkey,
        )
    )
    assert len(after_commitments) == len(existing_commitments)


def test_live_chain_fetch_submissions() -> None:
    """Fetch live revealed miner runtime submissions from chain."""

    require_live("CHRONOSEEK_LIVE_CHAIN", "NETWORK", "NETUID")

    subtensor, metagraph = live_subtensor_metagraph()
    snapshot = asyncio.run(
        load_chain_submission_snapshot(
            subtensor=subtensor,
            netuid=int(env_value("NETUID", str(DEFAULT_NETUID))),
            metagraph=metagraph,
        )
    )
    submissions = snapshot.submissions
    assert isinstance(submissions, dict)
    assert isinstance(snapshot.duplicate_hotkeys, set)

    expected_hotkey = env_value("LIVE_CHAIN_EXPECTED_HOTKEY")
    if expected_hotkey:
        assert expected_hotkey in submissions

    expected_duplicate_hotkey = env_value("LIVE_CHAIN_EXPECTED_DUPLICATE_HOTKEY")
    if expected_duplicate_hotkey:
        assert expected_duplicate_hotkey in snapshot.duplicate_hotkeys
        assert expected_duplicate_hotkey not in submissions
