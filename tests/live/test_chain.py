"""Live chain commit/fetch checks."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from chronoseek.chain.submissions import (
    MinerSubmission,
    commit_miner_submission,
    load_chain_submissions,
)
from live_helpers import env_value, require_extra_confirmation, require_live


pytestmark = pytest.mark.live


def live_subtensor_metagraph():
    import bittensor as bt

    subtensor = bt.Subtensor(network=env_value("NETWORK", "finney"))
    metagraph = bt.Metagraph(
        netuid=int(env_value("NETUID", "1")),
        network=subtensor.network,
        sync=False,
    )
    metagraph.sync(subtensor=subtensor)
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
            netuid=int(env_value("NETUID", "1")),
            submission=submission,
            blocks_until_reveal=int(env_value("LIVE_CHAIN_BLOCKS_UNTIL_REVEAL", "1")),
        )
    )
    assert success is True


def test_live_chain_fetch_submissions() -> None:
    """Fetch live revealed miner runtime submissions from chain."""

    require_live("CHRONOSEEK_LIVE_CHAIN", "NETWORK", "NETUID")

    subtensor, metagraph = live_subtensor_metagraph()
    submissions = asyncio.run(
        load_chain_submissions(
            subtensor=subtensor,
            netuid=int(env_value("NETUID", "1")),
            metagraph=metagraph,
        )
    )
    assert isinstance(submissions, dict)

    expected_hotkey = env_value("LIVE_CHAIN_EXPECTED_HOTKEY")
    if expected_hotkey:
        assert expected_hotkey in submissions
