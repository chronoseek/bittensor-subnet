import os
import sys
from unittest.mock import patch

from validator import get_config


def test_validator_config_defaults_to_finney_netuid_20():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.subtensor.network == "finney"
    assert config.netuid == 20


def test_validator_config_accepts_explicit_chain_overrides():
    with patch.dict(
        os.environ,
        {"NETWORK": "test", "NETUID": "298"},
        clear=True,
    ):
        with patch.object(
            sys,
            "argv",
            [
                "validator.py",
                "--subtensor.network",
                "local",
                "--netuid",
                "123",
            ],
        ):
            config = get_config()

    assert config.subtensor.network == "local"
    assert config.netuid == 123


def test_validator_submission_enforcement_defaults_to_enabled():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.enforce_one_hotkey_one_submission is True


def test_validator_submission_enforcement_can_be_disabled_by_environment():
    with patch.dict(
        os.environ,
        {"ENFORCE_ONE_HOTKEY_ONE_SUBMISSION": "false"},
        clear=True,
    ):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.enforce_one_hotkey_one_submission is False


def test_validator_vidaio_defaults_to_public_api():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.vidaio_compression_enabled is True
    assert config.vidaio_api_base_url == "https://api.vidaio.io"
    assert config.vidaio_poll_interval_seconds == 15.0


def test_validator_vidaio_reads_environment_defaults():
    with patch.dict(
        os.environ,
        {
            "VIDAIO_COMPRESSION_ENABLED": "0",
            "VIDAIO_API_BASE_URL": "https://vidaio.example",
            "VIDAIO_TIMEOUT_SECONDS": "17.5",
            "VIDAIO_POLL_INTERVAL_SECONDS": "22.5",
        },
        clear=True,
    ):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.vidaio_compression_enabled is False
    assert config.vidaio_api_base_url == "https://api.vidaio.io"
    assert config.vidaio_timeout_seconds == 17.5
    assert config.vidaio_poll_interval_seconds == 22.5


def test_validator_vidaio_base_url_accepts_cli_override():
    with patch.dict(
        os.environ,
        {"VIDAIO_API_BASE_URL": "https://env.example"},
        clear=True,
    ):
        with patch.object(
            sys,
            "argv",
            ["validator.py", "--vidaio-api-base-url", "https://cli.example"],
        ):
            config = get_config()

    assert config.vidaio_api_base_url == "https://cli.example"


def test_validator_vidaio_disable_flag_overrides_default():
    with patch.dict(os.environ, {"VIDAIO_COMPRESSION_ENABLED": "1"}, clear=True):
        with patch.object(
            sys,
            "argv",
            ["validator.py", "--disable-vidaio-compression"],
        ):
            config = get_config()

    assert config.vidaio_compression_enabled is False
