"""
ChronoSeek miner submission command.

Miners deploy their retrieval runtime on Chutes, then use this command to commit
the runtime metadata on-chain. This process does not serve HTTP locally.
"""

import argparse
import asyncio
import json
import os
import sys

import bittensor as bt
from dotenv import load_dotenv

from chronoseek.chain.submissions import (
    DuplicateMinerSubmissionError,
    MinerSubmission,
    commit_miner_submission,
    get_hotkey_revealed_commitments,
)
from chronoseek.constants import (
    DEFAULT_HOTKEY_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MECHID,
    DEFAULT_NETUID,
    DEFAULT_NETWORK,
    DEFAULT_WALLET_NAME,
    DEFAULT_WALLET_PATH,
)

load_dotenv()


class SubnetNotFoundError(RuntimeError):
    """Raised when the configured subnet is missing on the selected chain."""


def enable_bittensor_cli_parsing() -> None:
    """Bittensor 10.x disables argparse parsing unless this env flag opts in."""

    os.environ["BT_NO_PARSE_CLI_ARGS"] = "false"


def get_config():
    enable_bittensor_cli_parsing()
    parser = argparse.ArgumentParser(description="Commit ChronoSeek miner metadata")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)

    parser.add_argument(
        "--network",
        dest="subtensor.network",
        type=str,
        default=os.getenv("NETWORK", DEFAULT_NETWORK),
        help="Alias for --subtensor.network. Usually finney, test, or local.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", str(DEFAULT_NETUID))),
        help="Subnet NetUID",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="",
        help="Optional direct HTTPS endpoint for the deployed Chutes runtime.",
    )
    parser.add_argument(
        "--chute-id",
        type=str,
        default="",
        help="Canonical Chutes deployment identifier.",
    )
    parser.add_argument(
        "--chute-slug",
        type=str,
        default="",
        help="Chutes slug used by validators to resolve https://{slug}.chutes.ai.",
    )
    parser.add_argument("--artifact-id", type=str, default="")
    parser.add_argument("--artifact-revision", type=str, default="")
    parser.add_argument("--artifact-digest", type=str, default="")
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Runtime capability. Can be provided multiple times.",
    )
    parser.add_argument(
        "--blocks-until-reveal",
        type=int,
        default=1,
        help="Commit-reveal delay in blocks.",
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Fetch and print revealed commitments for this wallet hotkey, then exit without submitting.",
    )

    parser.set_defaults(
        **{
            "wallet.name": os.getenv("WALLET_NAME", DEFAULT_WALLET_NAME),
            "wallet.hotkey": os.getenv("HOTKEY_NAME", DEFAULT_HOTKEY_NAME),
            "wallet.path": os.getenv("WALLET_PATH", DEFAULT_WALLET_PATH),
            "subtensor.network": os.getenv("NETWORK", DEFAULT_NETWORK),
            "logging.level": os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        }
    )
    return bt.Config(parser)


def configure_logging(config) -> None:
    bt.logging(config=config, logging_dir=config.logging.logging_dir)
    bt.logging.on()
    if config.logging.level == "DEBUG":
        bt.logging.set_debug(True)
    elif config.logging.level == "TRACE":
        bt.logging.set_trace(True)
    else:
        bt.logging.set_info(True)


def get_wallet_hotkey_address(wallet) -> str | None:
    hotkey = getattr(wallet, "hotkey", None)
    return getattr(hotkey, "ss58_address", None)


def optional_text(value) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def load_subtensor_and_metagraph(config):
    netuid = int(config.netuid)
    mechid = DEFAULT_MECHID
    subtensor = bt.Subtensor(config=config)
    network = optional_text(getattr(subtensor, "network", None))
    chain_endpoint = optional_text(getattr(subtensor, "chain_endpoint", None))
    bt.logging.info(
        "Loading metagraph for "
        f"network={network}, "
        f"chain_endpoint={chain_endpoint}, "
        f"netuid={netuid}, mechid={mechid} "
        f"(Bittensor subnet mechanism {netuid}.{mechid})"
    )
    if hasattr(subtensor, "subnet_exists") and not subtensor.subnet_exists(
        netuid=netuid
    ):
        raise SubnetNotFoundError(
            f"Subnet netuid={netuid} does not exist on network={network} "
            f"({chain_endpoint}). Bittensor reports subnet mechanisms as "
            f"netuid.mechid, so {netuid}.{mechid} means netuid={netuid}, "
            f"mechid={mechid}. ChronoSeek uses mechid=0. If this is a "
            "testnet subnet, rerun with --network test."
        )

    metagraph = bt.Metagraph(
        netuid=netuid,
        mechid=mechid,
        network=subtensor.network,
        sync=False,
    )
    metagraph.sync(subtensor=subtensor)
    return subtensor, metagraph


def assert_registered_hotkey(wallet_hotkey: str, metagraph, netuid: int) -> bool:
    if wallet_hotkey not in metagraph.hotkeys:
        bt.logging.error(
            f"Miner hotkey {wallet_hotkey} is NOT registered on netuid {netuid}"
        )
        return False

    bt.logging.info(
        f"Miner registered with UID: {metagraph.hotkeys.index(wallet_hotkey)}"
    )
    return True


def build_submission_payload(config, hotkey: str) -> MinerSubmission:
    if not config.endpoint and not config.chute_slug:
        raise ValueError(
            "current validators require --endpoint or --chute-slug to resolve the runtime; --chute-id alone is not routable yet"
        )

    submission = MinerSubmission(
        hotkey=hotkey,
        endpoint=config.endpoint or None,
        chute_id=config.chute_id or None,
        chute_slug=config.chute_slug or None,
        artifact_id=config.artifact_id or None,
        artifact_revision=config.artifact_revision or None,
        artifact_digest=config.artifact_digest or None,
        capabilities=list(config.capability or []),
    )
    return submission


def normalize_commit_data(commit_data):
    if isinstance(commit_data, str):
        try:
            return json.loads(commit_data)
        except json.JSONDecodeError:
            return commit_data
    return commit_data


async def check_existing_commitments(config, subtensor, metagraph, wallet_hotkey: str):
    commits = await get_hotkey_revealed_commitments(
        subtensor=subtensor,
        netuid=int(config.netuid),
        hotkey=wallet_hotkey,
    )
    result = {
        "hotkey": wallet_hotkey,
        "netuid": int(config.netuid),
        "registered": wallet_hotkey in getattr(metagraph, "hotkeys", []),
        "commitments": [
            {
                "block": block,
                "data": normalize_commit_data(commit_data),
            }
            for block, commit_data in commits
        ],
    }

    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not commits:
        bt.logging.info(f"No revealed miner commitments found for {wallet_hotkey}.")
    elif len(commits) > 1:
        bt.logging.warning(
            f"Found {len(commits)} revealed miner commitments for {wallet_hotkey}; validators disqualify duplicate-submitted hotkeys."
        )
    else:
        bt.logging.info(f"Found one revealed miner commitment for {wallet_hotkey}.")
    return 0


async def submit_runtime_metadata(config) -> int:
    wallet = bt.Wallet(config=config)
    wallet_hotkey = get_wallet_hotkey_address(wallet)
    if not wallet_hotkey:
        bt.logging.error(
            "Wallet hotkey is unavailable. Check WALLET_NAME, HOTKEY_NAME, and WALLET_PATH."
        )
        return 1

    try:
        subtensor, metagraph = load_subtensor_and_metagraph(config)
    except SubnetNotFoundError as exc:
        bt.logging.error(str(exc))
        return 1

    if config.check_existing:
        return await check_existing_commitments(
            config=config,
            subtensor=subtensor,
            metagraph=metagraph,
            wallet_hotkey=wallet_hotkey,
        )

    if not assert_registered_hotkey(wallet_hotkey, metagraph, int(config.netuid)):
        return 1

    try:
        submission = build_submission_payload(config, wallet_hotkey)
    except Exception as exc:
        bt.logging.error(f"Invalid miner submission metadata: {exc}")
        bt.logging.error(
            "Provide --endpoint or --chute-slug for the deployed runtime. --chute-id is metadata-only until validators can resolve it through Chutes."
        )
        return 1

    payload = submission.model_dump(mode="json", exclude_none=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    bt.logging.info(
        f"Committing ChronoSeek v2 runtime metadata for {wallet_hotkey} on netuid={config.netuid}"
    )
    try:
        success = await commit_miner_submission(
            subtensor=subtensor,
            wallet=wallet,
            netuid=int(config.netuid),
            submission=submission,
            blocks_until_reveal=int(config.blocks_until_reveal),
        )
    except DuplicateMinerSubmissionError as exc:
        bt.logging.error(str(exc))
        return 1

    if not success:
        bt.logging.error("Chain rejected v2 miner submission commitment.")
        return 1

    bt.logging.success("ChronoSeek v2 miner submission committed.")
    return 0


def main():
    config = get_config()
    configure_logging(config)
    return asyncio.run(submit_runtime_metadata(config))


if __name__ == "__main__":
    sys.exit(main())
