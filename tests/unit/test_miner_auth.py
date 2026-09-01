import bittensor as bt

from chronoseek.miner.auth import (
    ValidatorAuthContext,
    authorize_hotkey,
    get_hotkey_stake,
    normalize_stake_value,
)


class DummyMetagraph:
    def __init__(self, hotkeys, stakes):
        self.hotkeys = hotkeys
        self.S = stakes


def test_normalize_stake_value_reads_alpha_balance_without_crashing():
    """Regression test: Balance.tao raises UnitMismatchError on a subnet
    (non-zero netuid) balance, and hasattr() only swallows AttributeError -
    so probing with hasattr(stake_value, "tao") propagated the exception and
    crashed every request. Same failure mode in reverse for .alpha on a
    netuid-0 (TAO) balance."""
    stake = bt.Balance.from_alpha(42.5, netuid=298)

    assert normalize_stake_value(stake) == 42.5


def test_normalize_stake_value_reads_tao_balance_without_crashing():
    stake = bt.Balance.from_tao(10.0)

    assert normalize_stake_value(stake) == 10.0


def test_normalize_stake_value_reads_numpy_like_item():
    class DummyNumpyScalar:
        def item(self):
            return 7.0

    assert normalize_stake_value(DummyNumpyScalar()) == 7.0


def test_normalize_stake_value_reads_plain_number():
    assert normalize_stake_value(3.0) == 3.0


def test_get_hotkey_stake_returns_matching_metagraph_stake():
    metagraph = DummyMetagraph(
        hotkeys=["hk1", "hk2"],
        stakes=[
            bt.Balance.from_alpha(10.0, netuid=298),
            bt.Balance.from_alpha(42.5, netuid=298),
        ],
    )

    assert get_hotkey_stake(metagraph, "hk2") == 42.5
    assert get_hotkey_stake(metagraph, "missing-hotkey") == 0.0


def test_authorize_hotkey_enforces_minimum_stake():
    auth_context = ValidatorAuthContext(
        min_validator_stake=20.0,
        metagraph=DummyMetagraph(
            hotkeys=["hk1", "hk2"],
            stakes=[
                bt.Balance.from_alpha(10.0, netuid=298),
                bt.Balance.from_alpha(42.5, netuid=298),
            ],
        ),
    )

    allowed, details = authorize_hotkey(auth_context, "hk2")
    assert allowed is True
    assert details["caller_stake"] == 42.5

    allowed, details = authorize_hotkey(auth_context, "hk1")
    assert allowed is False
    assert details["minimum_validator_stake"] == 20.0
