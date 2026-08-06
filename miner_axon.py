"""
ChronoSeek miner axon serving track (issue #32).

An alternative to Chutes deployment: this process serves the exact same
FastAPI contract as the Chutes runtime (`chronoseek.miner.runtime:app`)
directly over uvicorn, and separately publishes this host's ip:port
on-chain via the `ServeAxon` intent so validators can resolve it as a
fallback when no Chutes commitment exists for this hotkey.

Bittensor 11 removed the old axon/dendrite networking stack (`bt.axon`) -
`ServeAxon` only writes the ip:port to chain storage, it does not run any
server itself. All requests are served by the existing
`chronoseek.miner.runtime` app, which already verifies Epistula-signed
requests the same way the Chutes deployment path does.

This process does not commit any metadata on-chain (see `miner.py` for
that); a validator falls back to the published axon address only when
this hotkey has no Chutes commitment resolvable at all.
"""

import argparse
import os

import bittensor as bt
import uvicorn
from bittensor.intents import ServeAxon
from dotenv import load_dotenv

from chronoseek.bittensor_sdk import (
    add_bittensor_arguments,
    create_subtensor,
    create_wallet,
    fetch_metagraph,
    parse_config,
)
from chronoseek.constants import (
    DEFAULT_AXON_HOST,
    DEFAULT_AXON_PORT,
    DEFAULT_HOTKEY_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_NETUID,
    DEFAULT_NETWORK,
    DEFAULT_WALLET_NAME,
    DEFAULT_WALLET_PATH,
)
from chronoseek.logging import configure_logging as configure_application_logging
from chronoseek.logging import logger
from chronoseek.miner.runtime import app as miner_app

load_dotenv()


def get_config():
    parser = argparse.ArgumentParser(
        description="Serve the ChronoSeek miner runtime over a Bittensor axon"
    )
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
        "--axon-port",
        type=int,
        default=int(os.getenv("AXON_PORT", str(DEFAULT_AXON_PORT))),
        help="Local port to bind and to publish on-chain by default.",
    )
    parser.add_argument(
        "--axon-external-ip",
        type=str,
        default=os.getenv("AXON_EXTERNAL_IP", "").strip() or None,
        help=(
            "Public IP to publish on-chain. Required: Bittensor 11 has no "
            "external-IP auto-detection, unlike the old bt.axon."
        ),
    )
    parser.add_argument(
        "--axon-external-port",
        type=int,
        default=(
            int(os.getenv("AXON_EXTERNAL_PORT"))
            if os.getenv("AXON_EXTERNAL_PORT", "").strip()
            else None
        ),
        help="Public port to publish on-chain, if different from --axon-port.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("AXON_HOST", DEFAULT_AXON_HOST),
        help="Local host/interface for uvicorn to bind.",
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
        filename="miner_axon.log",
    )


def _find_own_neuron(metagraph, hotkey: str):
    hotkeys = list(getattr(metagraph, "hotkeys", None) or [])
    if hotkey not in hotkeys:
        return None
    uid = hotkeys.index(hotkey)
    neurons = getattr(metagraph, "neurons", None) or []
    return neurons[uid] if uid < len(neurons) else None


def _is_already_published(metagraph, hotkey: str, ip: str, port: int) -> bool:
    neuron = _find_own_neuron(metagraph, hotkey)
    published = getattr(neuron, "axon", None) if neuron is not None else None
    return bool(published) and str(published).strip() == f"{ip}:{port}"


def publish_axon(config, wallet: bt.Wallet, subtensor: bt.Subtensor, metagraph) -> None:
    if not config.axon_external_ip:
        raise ValueError(
            "--axon-external-ip (or AXON_EXTERNAL_IP) is required: Bittensor 11 "
            "has no external-IP auto-detection, so the public IP must be supplied "
            "explicitly for validators to resolve this axon."
        )

    ip = str(config.axon_external_ip)
    external_port = int(config.axon_external_port or config.axon_port)

    # A restart re-runs this on every process start; re-submitting the same
    # ip:port that's already on chain just burns the serving rate limit for
    # no effect, and can make a restart fail if that limit is exhausted.
    if _is_already_published(metagraph, wallet.hotkey.ss58_address, ip, external_port):
        logger.info(
            f"Axon already published on-chain at {ip}:{external_port} "
            f"netuid={config.netuid}; skipping republish."
        )
        return

    intent = ServeAxon(
        netuid=int(config.netuid),
        ip=ip,
        port=external_port,
    )
    result = subtensor.execute(intent, wallet)
    if not result.success:
        raise RuntimeError(f"Failed to publish axon on-chain: {result.message}")

    logger.success(
        f"Published axon on-chain: {ip}:{external_port} netuid={config.netuid}"
    )


def main() -> None:
    config = get_config()
    configure_logging(config)

    wallet = create_wallet(config)
    subtensor = create_subtensor(config)
    logger.info(
        f"Starting ChronoSeek miner axon runtime on network={config.subtensor.network}, "
        f"netuid={config.netuid}, hotkey={wallet.hotkey.ss58_address}"
    )

    metagraph = fetch_metagraph(subtensor, config.netuid)
    publish_axon(config, wallet, subtensor, metagraph)

    logger.info(f"Serving miner runtime on {config.host}:{config.axon_port}")
    uvicorn.run(miner_app, host=config.host, port=int(config.axon_port))


if __name__ == "__main__":
    main()
