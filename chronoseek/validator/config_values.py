from typing import Any


def get_config_float(config: Any, name: str, default: float) -> float:
    value = getattr(config, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_config_str(config: Any, name: str, default: str = "") -> str:
    value = getattr(config, name, default)
    return value if isinstance(value, str) else default


def get_config_bool(config: Any, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default
