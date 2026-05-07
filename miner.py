"""
ChronoSeek miner submission command.

Miners deploy their retrieval runtime on Chutes, then use this command to commit
the runtime metadata on-chain. This process does not serve HTTP locally.
"""

import argparse
import asyncio
import inspect
import json
import os
import sys

import bittensor as bt
from dotenv import load_dotenv

from chronoseek.validator.submissions import MinerSubmission

load_dotenv()


def get_config():
    parser = argparse.ArgumentParser(description="Commit ChronoSeek miner metadata")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)

    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", "1")),
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

    parser.set_defaults(
        **{
            "wallet.name": os.getenv("WALLET_NAME", "default"),
            "wallet.hotkey": os.getenv("HOTKEY_NAME", "default"),
            "wallet.path": os.getenv("WALLET_PATH", "~/.bittensor/wallets/"),
            "subtensor.network": os.getenv("NETWORK", "finney"),
            "logging.level": os.getenv("LOG_LEVEL", "INFO"),
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


async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def get_wallet_hotkey_address(wallet) -> str | None:
    hotkey = getattr(wallet, "hotkey", None)
    return getattr(hotkey, "ss58_address", None)


def load_subtensor_and_metagraph(config):
    subtensor = bt.Subtensor(config=config)
    metagraph = bt.Metagraph(
        netuid=config.netuid,
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


def build_submission_payload(config, hotkey: str) -> tuple[MinerSubmission, str]:
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
    payload = submission.model_dump(mode="json", exclude_none=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return submission, data


async def submit_runtime_metadata(config) -> int:
    wallet = bt.Wallet(config=config)
    wallet_hotkey = get_wallet_hotkey_address(wallet)
    if not wallet_hotkey:
        bt.logging.error(
            "Wallet hotkey is unavailable. Check WALLET_NAME, HOTKEY_NAME, and WALLET_PATH."
        )
        return 1

    subtensor, metagraph = load_subtensor_and_metagraph(config)
    if not assert_registered_hotkey(wallet_hotkey, metagraph, int(config.netuid)):
        return 1

    try:
        submission, data = build_submission_payload(config, wallet_hotkey)
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
    result = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=config.netuid,
        data=data,
        blocks_until_reveal=int(config.blocks_until_reveal),
    )
    response = await maybe_await(result)
    success = bool(getattr(response, "success", response))
    if not success:
        message = getattr(response, "message", None) or getattr(response, "error", None)
        bt.logging.error("Chain rejected v2 miner submission commitment.")
        if message:
            bt.logging.error(str(message))
        return 1

    bt.logging.success("ChronoSeek v2 miner submission committed.")
    return 0


def main():
    config = get_config()
    configure_logging(config)
    return asyncio.run(submit_runtime_metadata(config))


if __name__ == "__main__":
    sys.exit(main())
