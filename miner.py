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

from dotenv import load_dotenv

from chronoseek.bittensor_sdk import (
    SubnetNotFoundError,
    add_bittensor_arguments,
    create_subtensor,
    create_wallet,
    fetch_metagraph,
    parse_config,
)
from chronoseek.chain.submissions import (
    DuplicateMinerSubmissionError,
    MinerSubmission,
    commit_miner_submission,
    get_hotkey_revealed_commitments,
)
from chronoseek.constants import (
    DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
    DEFAULT_HOTKEY_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MECHID,
    DEFAULT_NETUID,
    DEFAULT_NETWORK,
    DEFAULT_WALLET_NAME,
    DEFAULT_WALLET_PATH,
)
from chronoseek.logging import configure_logging as configure_application_logging
from chronoseek.logging import logger

load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def get_config():
    parser = argparse.ArgumentParser(description="Commit ChronoSeek miner metadata")
    add_bittensor_arguments(parser)

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
    parser.add_argument(
        "--enforce-one-hotkey-one-submission",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENFORCE_ONE_HOTKEY_ONE_SUBMISSION",
            DEFAULT_ENFORCE_ONE_HOTKEY_ONE_SUBMISSION,
        ),
        help="Reject repeat submissions for a hotkey. Enabled by default.",
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
    return parse_config(parser)


def configure_logging(config) -> None:
    configure_application_logging(
        config.logging.level,
        logging_dir=config.logging.logging_dir,
        filename="miner.log",
    )


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
    subtensor = create_subtensor(config)
    network = optional_text(getattr(subtensor, "network", None))
    chain_endpoint = optional_text(getattr(subtensor, "endpoint", None))
    logger.info(
        "Loading metagraph for "
        f"network={network}, "
        f"chain_endpoint={chain_endpoint}, "
        f"netuid={netuid}, mechid={mechid} "
        f"(Bittensor subnet mechanism {netuid}.{mechid})"
    )
    try:
        metagraph = fetch_metagraph(subtensor, netuid)
    except SubnetNotFoundError as exc:
        raise SubnetNotFoundError(
            f"Subnet netuid={netuid} does not exist on network={network} "
            f"({chain_endpoint}). Bittensor reports subnet mechanisms as "
            f"netuid.mechid, so {netuid}.{mechid} means netuid={netuid}, "
            f"mechid={mechid}. ChronoSeek uses mechid=0. If this is a "
            "testnet subnet, rerun with --network test."
        ) from exc
    return subtensor, metagraph


def assert_registered_hotkey(wallet_hotkey: str, metagraph, netuid: int) -> bool:
    if wallet_hotkey not in metagraph.hotkeys:
        logger.error(
            f"Miner hotkey {wallet_hotkey} is NOT registered on netuid {netuid}"
        )
        return False

    logger.info(
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
        logger.info(f"No revealed miner commitments found for {wallet_hotkey}.")
    elif len(commits) > 1:
        if config.enforce_one_hotkey_one_submission:
            logger.warning(
                f"Found {len(commits)} revealed miner commitments for {wallet_hotkey}; validators disqualify duplicate-submitted hotkeys."
            )
        else:
            logger.info(
                f"Found {len(commits)} revealed miner commitments for {wallet_hotkey}; enforcement is disabled and validators use the newest submission."
            )
    else:
        logger.info(f"Found one revealed miner commitment for {wallet_hotkey}.")
    return 0


async def submit_runtime_metadata(config) -> int:
    wallet = create_wallet(config)
    wallet_hotkey = get_wallet_hotkey_address(wallet)
    if not wallet_hotkey:
        logger.error(
            "Wallet hotkey is unavailable. Check WALLET_NAME, HOTKEY_NAME, and WALLET_PATH."
        )
        return 1

    try:
        subtensor, metagraph = load_subtensor_and_metagraph(config)
    except SubnetNotFoundError as exc:
        logger.error(str(exc))
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
        logger.error(f"Invalid miner submission metadata: {exc}")
        logger.error(
            "Provide --endpoint or --chute-slug for the deployed runtime. --chute-id is metadata-only until validators can resolve it through Chutes."
        )
        return 1

    payload = submission.model_dump(mode="json", exclude_none=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(
        f"Committing ChronoSeek v2 runtime metadata for {wallet_hotkey} on netuid={config.netuid}"
    )
    try:
        success = await commit_miner_submission(
            subtensor=subtensor,
            wallet=wallet,
            netuid=int(config.netuid),
            submission=submission,
            blocks_until_reveal=int(config.blocks_until_reveal),
            enforce_one_hotkey_one_submission=bool(
                config.enforce_one_hotkey_one_submission
            ),
        )
    except DuplicateMinerSubmissionError as exc:
        logger.error(str(exc))
        return 1

    if not success:
        logger.error("Chain rejected v2 miner submission commitment.")
        return 1

    logger.success("ChronoSeek v2 miner submission committed.")
    return 0


def main():
    config = get_config()
    configure_logging(config)
    return asyncio.run(submit_runtime_metadata(config))


if __name__ == "__main__":
    sys.exit(main())
