"""Shared helpers for opt-in live integration tests."""

from __future__ import annotations

import os

import pytest


LIVE_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in LIVE_TRUE_VALUES


def env_value(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


def require_live(module_flag: str, *required_env: str) -> None:
    if not env_bool("CHRONOSEEK_LIVE_TESTS"):
        pytest.skip("set CHRONOSEEK_LIVE_TESTS=1 to enable live integration tests")
    if not env_bool(module_flag):
        pytest.skip(f"set {module_flag}=1 to run this module's live tests")

    missing = [name for name in required_env if not env_value(name)]
    if missing:
        pytest.skip(f"missing required env for live test: {', '.join(missing)}")


def require_extra_confirmation(flag: str, reason: str) -> None:
    if not env_bool(flag):
        pytest.skip(f"set {flag}=1 to confirm live {reason}")
