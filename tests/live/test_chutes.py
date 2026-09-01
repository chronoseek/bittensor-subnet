"""Live Chutes build/deploy checks."""

from __future__ import annotations

import asyncio

import pytest

from chronoseek.chutes.deployment import build_image_via_api, deploy_chute_via_api
from live_helpers import (
    env_bool,
    env_value,
    require_extra_confirmation,
    require_live,
)


pytestmark = pytest.mark.live


def test_live_chutes_build() -> None:
    """Start a live Chutes image build through the Chutes API."""

    require_live("CHRONOSEEK_LIVE_CHUTES", "CHUTES_API_KEY")
    require_extra_confirmation("CHRONOSEEK_LIVE_CHUTES_BUILD", "Chutes image build")

    payload = asyncio.run(
        build_image_via_api(
            api_base_url=env_value("LIVE_CHUTES_API_BASE_URL", "https://api.chutes.ai"),
            chute_ref=env_value("LIVE_CHUTES_CHUTE_REF", "chronoseek_chute:chute"),
            include_cwd=env_bool("LIVE_CHUTES_INCLUDE_CWD"),
            public=env_bool("LIVE_CHUTES_PUBLIC"),
            overwrite_existing=env_bool("LIVE_CHUTES_OVERWRITE_EXISTING"),
            timeout_seconds=float(env_value("LIVE_CHUTES_TIMEOUT_SECONDS", "900")),
            chute_name=env_value("LIVE_CHUTES_CHUTE_NAME") or None,
            chute_slug=env_value("LIVE_CHUTES_CHUTE_SLUG") or None,
            chute_display_name=env_value("LIVE_CHUTES_DISPLAY_NAME") or None,
        )
    )
    assert isinstance(payload, dict)
    assert payload


def test_live_chutes_deploy() -> None:
    """Deploy a live Chutes runtime through the Chutes API."""

    require_live("CHRONOSEEK_LIVE_CHUTES", "CHUTES_API_KEY")
    require_extra_confirmation("CHRONOSEEK_LIVE_CHUTES_DEPLOY", "Chutes deploy")
    require_extra_confirmation("CHRONOSEEK_LIVE_CHUTES_ACCEPT_FEE", "Chutes fee accept")

    payload = asyncio.run(
        deploy_chute_via_api(
            api_base_url=env_value("LIVE_CHUTES_API_BASE_URL", "https://api.chutes.ai"),
            chute_ref=env_value("LIVE_CHUTES_CHUTE_REF", "chronoseek_chute:chute"),
            accept_fee=True,
            public=env_bool("LIVE_CHUTES_PUBLIC"),
            timeout_seconds=float(env_value("LIVE_CHUTES_TIMEOUT_SECONDS", "900")),
            chute_name=env_value("LIVE_CHUTES_CHUTE_NAME") or None,
            chute_slug=env_value("LIVE_CHUTES_CHUTE_SLUG") or None,
            chute_display_name=env_value("LIVE_CHUTES_DISPLAY_NAME") or None,
        )
    )
    assert isinstance(payload, dict)
    assert payload
