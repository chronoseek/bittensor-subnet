import os
import sys
from unittest.mock import patch

from validator import get_config


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


def test_validator_hippius_s3_defaults_when_env_unset():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.hippius_s3_bucket == "chronoseek"


def test_validator_hippius_s3_reads_environment_defaults():
    """Regression test (PR #33 review): none of the --hippius-s3-* args read
    their corresponding env vars, so .env's HIPPIUS_S3_BUCKET was silently
    ignored and the validator authenticated against the wrong bucket."""
    with patch.dict(
        os.environ,
        {
            "HIPPIUS_S3_ENDPOINT_URL": "https://s3.example.com",
            "HIPPIUS_S3_PUBLIC_BASE_URL": "https://public.example.com",
            "HIPPIUS_S3_BUCKET": "chronoseek85",
            "HIPPIUS_S3_REGION": "example-region",
            "HIPPIUS_S3_TIMEOUT_SECONDS": "42.5",
        },
        clear=True,
    ):
        with patch.object(sys, "argv", ["validator.py"]):
            config = get_config()

    assert config.hippius_s3_endpoint_url == "https://s3.example.com"
    assert config.hippius_s3_public_base_url == "https://public.example.com"
    assert config.hippius_s3_bucket == "chronoseek85"
    assert config.hippius_s3_region == "example-region"
    assert config.hippius_s3_timeout_seconds == 42.5


def test_validator_hippius_s3_bucket_accepts_cli_override():
    with patch.dict(os.environ, {"HIPPIUS_S3_BUCKET": "env-bucket"}, clear=True):
        with patch.object(
            sys,
            "argv",
            ["validator.py", "--hippius-s3-bucket", "cli-bucket"],
        ):
            config = get_config()

    assert config.hippius_s3_bucket == "cli-bucket"
