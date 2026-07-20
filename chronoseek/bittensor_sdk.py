"""ChronoSeek's boundary around the Bittensor 11 SDK."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import bittensor as bt


class SubnetNotFoundError(RuntimeError):
    """Raised when a configured subnet does not exist on the selected chain."""


class MetagraphSnapshot:
    """Expose the Bittensor 11 metagraph through ChronoSeek's domain fields."""

    def __init__(self, raw: Any, *, network: str | None = None) -> None:
        self.raw = raw
        self.network = network

    @property
    def n(self) -> int:
        return len(self.raw.neurons)

    @property
    def uids(self) -> list[int]:
        return [int(neuron.uid) for neuron in self.raw.neurons]

    @property
    def hotkeys(self) -> list[str]:
        return list(self.raw.hotkeys)

    @property
    def coldkeys(self) -> list[str]:
        return list(self.raw.coldkeys)

    @property
    def I(self) -> list[float]:
        return [float(neuron.incentive) for neuron in self.raw.neurons]

    @property
    def S(self) -> list[Any]:
        return [neuron.total_stake for neuron in self.raw.neurons]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    def __len__(self) -> int:
        return len(self.raw.neurons)


def add_bittensor_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the SDK-related options previously supplied by Bittensor."""

    parser.add_argument("--wallet.name", dest="wallet.name", type=str)
    parser.add_argument("--wallet.hotkey", dest="wallet.hotkey", type=str)
    parser.add_argument("--wallet.path", dest="wallet.path", type=str)
    parser.add_argument("--subtensor.network", dest="subtensor.network", type=str)
    parser.add_argument(
        "--subtensor.chain-endpoint",
        "--subtensor.chain_endpoint",
        dest="subtensor.chain_endpoint",
        type=str,
        default="",
        help="Optional WebSocket chain endpoint; overrides --subtensor.network.",
    )
    parser.add_argument("--logging.level", dest="logging.level", type=str)
    parser.add_argument(
        "--logging.logging-dir",
        "--logging.logging_dir",
        dest="logging.logging_dir",
        type=str,
        default="",
    )


def parse_config(parser: argparse.ArgumentParser) -> SimpleNamespace:
    """Parse argparse values and restore dotted options as nested namespaces."""

    flat_config = parser.parse_args()
    config = SimpleNamespace()
    for dotted_name, value in vars(flat_config).items():
        target = config
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            child = getattr(target, part, None)
            if child is None:
                child = SimpleNamespace()
                setattr(target, part, child)
            target = child
        setattr(target, parts[-1], value)
    return config


def create_wallet(config: Any) -> bt.Wallet:
    return bt.Wallet(
        name=str(config.wallet.name),
        hotkey=str(config.wallet.hotkey),
        path=str(Path(config.wallet.path).expanduser()),
    )


def create_subtensor(config: Any) -> bt.Subtensor:
    endpoint = str(getattr(config.subtensor, "chain_endpoint", "") or "").strip()
    network = endpoint or str(config.subtensor.network)
    return bt.Subtensor(network=network)


def fetch_metagraph(
    subtensor: Any,
    netuid: int,
    *,
    commitments: bool = False,
) -> MetagraphSnapshot:
    raw_metagraph = subtensor.subnets.metagraph(
        int(netuid),
        commitments=commitments,
    )
    if raw_metagraph is None:
        raise SubnetNotFoundError(
            f"Subnet netuid={int(netuid)} does not exist on network="
            f"{getattr(subtensor, 'network', 'unknown')} "
            f"({getattr(subtensor, 'endpoint', 'unknown')})."
        )
    return MetagraphSnapshot(raw_metagraph, network=getattr(subtensor, "network", None))


def current_block(subtensor: Any) -> int:
    return int(subtensor.block())


def subnet_tempo(subtensor: Any, netuid: int) -> int:
    return int(subtensor.subnets.info(int(netuid)).tempo)


def set_weights(
    *,
    subtensor: Any,
    wallet: bt.Wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    mechid: int = 0,
) -> bool:
    result = subtensor.execute(
        bt.SetWeights(
            netuid=int(netuid),
            uids=[int(uid) for uid in uids],
            weights=[float(weight) for weight in weights],
            mechid=int(mechid),
        ),
        wallet,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    return bool(result)
