from chronoseek.miner.auth import (
    ValidatorAuthContext,
    authorize_hotkey,
    get_hotkey_stake,
)


class DummyStake:
    def __init__(self, tao):
        self.tao = tao


class DummyMetagraph:
    def __init__(self, hotkeys, stakes):
        self.hotkeys = hotkeys
        self.S = stakes


def test_get_hotkey_stake_returns_matching_metagraph_stake():
    metagraph = DummyMetagraph(
        hotkeys=["hk1", "hk2"],
        stakes=[DummyStake(10.0), DummyStake(42.5)],
    )

    assert get_hotkey_stake(metagraph, "hk2") == 42.5
    assert get_hotkey_stake(metagraph, "missing-hotkey") == 0.0


def test_authorize_hotkey_enforces_minimum_stake():
    auth_context = ValidatorAuthContext(
        min_validator_stake=20.0,
        metagraph=DummyMetagraph(
            hotkeys=["hk1", "hk2"],
            stakes=[DummyStake(10.0), DummyStake(42.5)],
        ),
    )

    allowed, details = authorize_hotkey(auth_context, "hk2")
    assert allowed is True
    assert details["caller_stake"] == 42.5

    allowed, details = authorize_hotkey(auth_context, "hk1")
    assert allowed is False
    assert details["minimum_validator_stake"] == 20.0
