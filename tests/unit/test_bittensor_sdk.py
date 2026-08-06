import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chronoseek.bittensor_sdk import (
    MetagraphSnapshot,
    add_bittensor_arguments,
    create_subtensor,
    create_wallet,
    current_block,
    fetch_metagraph,
    parse_config,
    set_weights,
    subnet_tempo,
)


def test_legacy_dotted_cli_spellings_remain_supported():
    parser = argparse.ArgumentParser()
    add_bittensor_arguments(parser)

    with patch(
        "sys.argv",
        [
            "validator.py",
            "--subtensor.chain_endpoint",
            "ws://127.0.0.1:9944",
            "--logging.logging_dir",
            "/tmp/chronoseek-logs",
        ],
    ):
        config = parse_config(parser)

    assert config.subtensor.chain_endpoint == "ws://127.0.0.1:9944"
    assert config.logging.logging_dir == "/tmp/chronoseek-logs"


def test_metagraph_snapshot_exposes_chronoseek_score_and_stake_fields():
    raw = SimpleNamespace(
        neurons=[
            SimpleNamespace(uid=0, incentive=0.25, total_stake="stake-0"),
            SimpleNamespace(uid=1, incentive=0.75, total_stake="stake-1"),
        ],
        hotkeys=["hk-0", "hk-1"],
        coldkeys=["ck-0", "ck-1"],
        tempo=100,
    )

    metagraph = MetagraphSnapshot(raw, network="test")

    assert metagraph.n == 2
    assert metagraph.uids == [0, 1]
    assert metagraph.hotkeys == ["hk-0", "hk-1"]
    assert metagraph.coldkeys == ["ck-0", "ck-1"]
    assert metagraph.I == [0.25, 0.75]
    assert metagraph.S == ["stake-0", "stake-1"]
    assert metagraph.tempo == 100


def test_fetch_metagraph_uses_bittensor_11_subnet_namespace():
    raw = SimpleNamespace(neurons=[], hotkeys=[], coldkeys=[])
    subtensor = MagicMock()
    subtensor.network = "test"
    subtensor.subnets.metagraph.return_value = raw

    metagraph = fetch_metagraph(subtensor, 298)

    assert metagraph.raw is raw
    subtensor.subnets.metagraph.assert_called_once_with(298, commitments=False)


def test_wallet_and_subtensor_use_explicit_constructor_arguments(tmp_path):
    config = SimpleNamespace(
        wallet=SimpleNamespace(name="alice", hotkey="validator", path=str(tmp_path)),
        subtensor=SimpleNamespace(network="test", chain_endpoint=""),
    )

    with patch("chronoseek.bittensor_sdk.bt.Wallet") as wallet_cls, patch(
        "chronoseek.bittensor_sdk.bt.Subtensor"
    ) as subtensor_cls:
        create_wallet(config)
        create_subtensor(config)

    wallet_cls.assert_called_once_with(
        name="alice",
        hotkey="validator",
        path=str(tmp_path),
    )
    subtensor_cls.assert_called_once_with(network="test")


def test_chain_reads_and_weight_intent_use_bittensor_11_api():
    subtensor = MagicMock()
    subtensor.block.return_value = 123
    subtensor.subnets.info.return_value.tempo = 360
    subtensor.execute.return_value = True
    wallet = MagicMock()

    assert current_block(subtensor) == 123
    assert subnet_tempo(subtensor, 1) == 360
    assert set_weights(
        subtensor=subtensor,
        wallet=wallet,
        netuid=1,
        uids=[0, 1],
        weights=[0.25, 0.75],
    )

    intent = subtensor.execute.call_args.args[0]
    assert intent.netuid == 1
    assert intent.uids == [0, 1]
    assert intent.weights == [0.25, 0.75]
    assert subtensor.execute.call_args.args[1] is wallet


def test_current_block_retries_past_transient_rpc_failure():
    subtensor = MagicMock()
    subtensor.block.side_effect = [RuntimeError("no header for None"), 123]

    with patch("chronoseek.bittensor_sdk.time.sleep") as sleep_mock:
        assert current_block(subtensor, retry_delay_seconds=0.0) == 123

    assert subtensor.block.call_count == 2
    sleep_mock.assert_called_once_with(0.0)


def test_current_block_raises_after_exhausting_retries():
    subtensor = MagicMock()
    subtensor.block.side_effect = RuntimeError("no header for None")

    with patch("chronoseek.bittensor_sdk.time.sleep"):
        try:
            current_block(subtensor, max_attempts=3, retry_delay_seconds=0.0)
        except RuntimeError as error:
            assert "no header for None" in str(error)
        else:
            raise AssertionError("expected current_block to raise")

    assert subtensor.block.call_count == 3
