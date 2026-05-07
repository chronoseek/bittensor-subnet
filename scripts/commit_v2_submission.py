#!/usr/bin/env python3
"""Commit a ChronoSeek v2 miner submission to the subnet.

This mirrors the Affine pattern: miners publish runtime metadata on-chain;
validators resolve it for synthetic evaluation. Secrets are never committed.

The canonical miner command is `python miner.py`; this script is retained as a
compatibility helper for operators who already adopted it.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Commit a ChronoSeek v2 submission")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", "1")),
        help="Subnet NetUID.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="",
        help="Optional direct HTTPS endpoint for the private runtime.",
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
    return parser


async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def main_async() -> int:
    config = bt.Config(build_parser())
    bt.logging(config=config, logging_dir=config.logging.logging_dir)
    bt.logging.on()
    bt.logging.set_info(True)

    wallet = bt.Wallet(config=config)
    if not wallet.hotkey:
        bt.logging.error("Wallet hotkey is required.")
        return 1

    if not config.endpoint and not config.chute_slug:
        bt.logging.error(
            "Current validators require --endpoint or --chute-slug to resolve the runtime; --chute-id alone is metadata-only until Chutes lookup is implemented."
        )
        return 1

    submission = MinerSubmission(
        hotkey=wallet.hotkey.ss58_address,
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

    print(json.dumps(payload, indent=2, sort_keys=True))

    bt.logging.info(
        f"Committing ChronoSeek v2 submission for {wallet.hotkey.ss58_address} on netuid={config.netuid}"
    )
    subtensor = bt.Subtensor(config=config)
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
        bt.logging.error("Chain rejected v2 submission commitment.")
        if message:
            bt.logging.error(str(message))
        return 1

    bt.logging.success("ChronoSeek v2 submission committed.")
    return 0


def main():
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
