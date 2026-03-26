from dataclasses import dataclass
from typing import Any


@dataclass
class ValidatorAuthContext:
    min_validator_stake: float
    metagraph: Any


def normalize_stake_value(stake_value: Any) -> float:
    if hasattr(stake_value, "tao"):
        return float(stake_value.tao)
    if hasattr(stake_value, "item"):
        return float(stake_value.item())
    return float(stake_value)


def get_hotkey_stake(metagraph: Any, hotkey: str) -> float:
    if metagraph is None:
        return 0.0

    try:
        uid = metagraph.hotkeys.index(hotkey)
    except ValueError:
        return 0.0

    stakes = getattr(metagraph, "S", None)
    if stakes is None or uid >= len(stakes):
        return 0.0

    return normalize_stake_value(stakes[uid])


def authorize_hotkey(
    auth_context: ValidatorAuthContext | None,
    hotkey: str,
) -> tuple[bool, dict[str, float | str]]:
    if auth_context is None:
        return False, {
            "caller_hotkey": hotkey,
            "caller_stake": 0.0,
            "minimum_validator_stake": 0.0,
        }

    caller_stake = get_hotkey_stake(auth_context.metagraph, hotkey)
    return caller_stake >= auth_context.min_validator_stake, {
        "caller_hotkey": hotkey,
        "caller_stake": caller_stake,
        "minimum_validator_stake": auth_context.min_validator_stake,
    }
