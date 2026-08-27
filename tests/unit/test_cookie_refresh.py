import base64
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from chronoseek.miner import cookie_refresh, cookie_refresh_server


def _launch_jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _write_registry(tmp_path: Path, cookie_file: Path, secret: str) -> Path:
    registry = tmp_path / "bindings.json"
    registry.write_text(
        json.dumps(
            {
                "bindings": {
                    "chute-a": {
                        "secret": secret,
                        "cookie_file": str(cookie_file),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_server_authenticates_binding_and_reloads_cookie_file(tmp_path, monkeypatch):
    secret = "a" * 64
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\nfirst\n", encoding="utf-8")
    registry = _write_registry(tmp_path, cookie_file, secret)
    monkeypatch.setenv(cookie_refresh_server.REGISTRY_PATH_ENV, str(registry))
    cookie_refresh_server._used_nonces.clear()

    def request(nonce: str):
        timestamp = str(int(time.time()))
        signature = cookie_refresh.sign_refresh_request(
            binding_id="chute-a", timestamp=timestamp, nonce=nonce, secret=secret
        )
        return cookie_refresh_server.refresh_cookies(
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            x_chronoseek_binding="chute-a",
            x_chronoseek_timestamp=timestamp,
            x_chronoseek_nonce=nonce,
            x_chronoseek_signature=signature,
        )

    first = request("1" * 32)
    assert b"first" in first.body

    cookie_file.write_text("# Netscape HTTP Cookie File\nsecond\n", encoding="utf-8")
    second = request("2" * 32)
    assert b"second" in second.body


def test_server_rejects_bad_signature_and_nonce_replay(tmp_path, monkeypatch):
    secret = "b" * 64
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    registry = _write_registry(tmp_path, cookie_file, secret)
    monkeypatch.setenv(cookie_refresh_server.REGISTRY_PATH_ENV, str(registry))
    cookie_refresh_server._used_nonces.clear()
    timestamp = str(int(time.time()))
    nonce = "3" * 32

    with pytest.raises(HTTPException) as bad_signature:
        cookie_refresh_server.refresh_cookies(
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            x_chronoseek_binding="chute-a",
            x_chronoseek_timestamp=timestamp,
            x_chronoseek_nonce=nonce,
            x_chronoseek_signature="0" * 64,
        )
    assert bad_signature.value.status_code == 403

    signature = cookie_refresh.sign_refresh_request(
        binding_id="chute-a", timestamp=timestamp, nonce=nonce, secret=secret
    )
    cookie_refresh_server.refresh_cookies(
        request=MagicMock(client=MagicMock(host="127.0.0.1")),
        x_chronoseek_binding="chute-a",
        x_chronoseek_timestamp=timestamp,
        x_chronoseek_nonce=nonce,
        x_chronoseek_signature=signature,
    )
    with pytest.raises(HTTPException) as replay:
        cookie_refresh_server.refresh_cookies(
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            x_chronoseek_binding="chute-a",
            x_chronoseek_timestamp=timestamp,
            x_chronoseek_nonce=nonce,
            x_chronoseek_signature=signature,
        )
    assert replay.value.status_code == 409


def test_client_requires_tee_and_writes_private_cookie_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUTES_EXECUTION_CONTEXT", "REMOTE")
    monkeypatch.setenv("CHUTES_LAUNCH_JWT", _launch_jwt({"env_type": "tee"}))
    monkeypatch.setenv(cookie_refresh.COOKIE_REFRESH_URL_ENV, "https://miner.example")
    monkeypatch.setenv(cookie_refresh.COOKIE_REFRESH_BINDING_ID_ENV, "chute-a")
    monkeypatch.setenv(cookie_refresh.COOKIE_REFRESH_SHARED_SECRET_ENV, "c" * 64)

    response = MagicMock()
    response.content = b"# Netscape HTTP Cookie File\nfresh\n"
    response.headers = {"X-Chronoseek-Cookie-Version": "abc123"}
    response.raise_for_status.return_value = None
    existing = tmp_path / "cookies.txt"
    existing.write_text("# Netscape HTTP Cookie File\nstale\n", encoding="utf-8")
    with patch.object(cookie_refresh.requests, "post", return_value=response) as post:
        result = Path(cookie_refresh.refresh_cookie_file(runtime_directory=str(tmp_path)))

    assert result == existing
    assert result.read_bytes() == response.content
    assert result.stat().st_mode & 0o777 == 0o600
    assert os.environ["YTDLP_COOKIES"] == str(existing.resolve())
    assert post.call_args.args[0] == "https://miner.example/v1/cookies/refresh"


def test_client_rejects_non_tee_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv(cookie_refresh.COOKIE_REFRESH_ALLOW_NON_TEE_ENV, raising=False)
    monkeypatch.setenv("CHUTES_EXECUTION_CONTEXT", "REMOTE")
    monkeypatch.setenv("CHUTES_LAUNCH_JWT", _launch_jwt({"env_type": "graval"}))
    with pytest.raises(RuntimeError, match="only inside a Chutes TEE"):
        cookie_refresh.refresh_cookie_file(runtime_directory=str(tmp_path))


def test_local_axon_client_does_not_require_chutes_launch_token(monkeypatch):
    monkeypatch.delenv("CHUTES_EXECUTION_CONTEXT", raising=False)
    monkeypatch.delenv("CHUTES_LAUNCH_JWT", raising=False)
    monkeypatch.delenv(cookie_refresh.COOKIE_REFRESH_ALLOW_NON_TEE_ENV, raising=False)
    assert cookie_refresh.is_refresh_client_allowed() is True
