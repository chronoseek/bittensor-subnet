"""Authenticated cookie refresh between a miner Chute and its operator endpoint."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path

import requests

from chronoseek.logging import logger


COOKIE_REFRESH_URL_ENV = "COOKIE_REFRESH_URL"
COOKIE_REFRESH_BINDING_ID_ENV = "COOKIE_REFRESH_BINDING_ID"
COOKIE_REFRESH_SHARED_SECRET_ENV = "COOKIE_REFRESH_SHARED_SECRET"
COOKIE_REFRESH_ALLOW_NON_TEE_ENV = "COOKIE_REFRESH_ALLOW_NON_TEE"
COOKIE_REFRESH_TIMEOUT_SECONDS_ENV = "COOKIE_REFRESH_TIMEOUT_SECONDS"
COOKIE_REFRESH_INTERVAL_SECONDS_ENV = "COOKIE_REFRESH_INTERVAL_SECONDS"
COOKIE_REFRESH_CACHE_PATH_ENV = "COOKIE_REFRESH_CACHE_PATH"
DEFAULT_COOKIE_REFRESH_CACHE_PATH = "/tmp/chronoseek-cookie-cache/cookies.txt"
COOKIE_REFRESH_PATH = "/v1/cookies/refresh"
_refresh_stop = threading.Event()
_refresh_thread: threading.Thread | None = None
_refresh_thread_lock = threading.Lock()


def signature_payload(*, binding_id: str, timestamp: str, nonce: str) -> bytes:
    return f"{binding_id}\n{timestamp}\n{nonce}".encode("utf-8")


def sign_refresh_request(
    *, binding_id: str, timestamp: str, nonce: str, secret: str
) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        signature_payload(binding_id=binding_id, timestamp=timestamp, nonce=nonce),
        hashlib.sha256,
    ).hexdigest()


def _launch_token_payload() -> dict:
    token = os.getenv("CHUTES_LAUNCH_JWT", "").strip()
    if not token:
        return {}
    try:
        encoded = token.split(".", 2)[1]
        encoded += "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def is_refresh_client_allowed() -> bool:
    """Allow local axon clients; require TEE for remote Chutes clients."""

    if os.getenv(COOKIE_REFRESH_ALLOW_NON_TEE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    is_remote_chutes = (
        os.getenv("CHUTES_EXECUTION_CONTEXT", "").strip().upper() == "REMOTE"
    )
    if not is_remote_chutes:
        return True
    return _launch_token_payload().get("env_type") == "tee"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for cookie refresh.")
    return value


def refresh_cookie_file(
    *,
    runtime_directory: str | None = None,
    target_path: str | None = None,
) -> str:
    """Fetch cookies and atomically replace the instance-persistent cache."""

    if not is_refresh_client_allowed():
        raise RuntimeError(
            "Remote cookie refresh is allowed only inside a Chutes TEE runtime."
        )

    base_url = _required_env(COOKIE_REFRESH_URL_ENV).rstrip("/")
    binding_id = _required_env(COOKIE_REFRESH_BINDING_ID_ENV)
    shared_secret = _required_env(COOKIE_REFRESH_SHARED_SECRET_ENV)
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    signature = sign_refresh_request(
        binding_id=binding_id,
        timestamp=timestamp,
        nonce=nonce,
        secret=shared_secret,
    )
    timeout = max(
        1.0,
        float(os.getenv(COOKIE_REFRESH_TIMEOUT_SECONDS_ENV, "15")),
    )
    response = requests.post(
        f"{base_url}{COOKIE_REFRESH_PATH}",
        headers={
            "X-Chronoseek-Binding": binding_id,
            "X-Chronoseek-Timestamp": timestamp,
            "X-Chronoseek-Nonce": nonce,
            "X-Chronoseek-Signature": signature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    cookie_bytes = response.content
    if not cookie_bytes.startswith(b"# Netscape HTTP Cookie File"):
        raise RuntimeError("Cookie refresh endpoint returned an invalid cookies.txt file.")

    configured_target = (
        target_path
        or os.getenv(COOKIE_REFRESH_CACHE_PATH_ENV, "").strip()
        or (str(Path(runtime_directory) / "cookies.txt") if runtime_directory else "")
        or DEFAULT_COOKIE_REFRESH_CACHE_PATH
    )
    destination = Path(configured_target).expanduser().resolve()
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix="cookies-", suffix=".txt", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(cookie_bytes)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
        os.chmod(destination, 0o600)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise

    print(
        "Refreshed yt-dlp cookies from the miner-bound endpoint "
        f"to {destination} (bytes={len(cookie_bytes)}, "
        f"version={response.headers.get('X-Chronoseek-Cookie-Version', 'unknown')})."
    )
    # Future downloader calls begin with the refreshed persistent copy. In a
    # Chutes runtime this replaces the writable startup copy, never the
    # read-only cookie baked into the image layer.
    os.environ["YTDLP_COOKIES"] = str(destination)
    return str(destination)


def configured_cookie_target() -> str | None:
    return (
        os.getenv("YTDLP_COOKIES", "").strip()
        or os.getenv(COOKIE_REFRESH_CACHE_PATH_ENV, "").strip()
        or None
    )


def _periodic_refresh_loop(interval_seconds: float) -> None:
    while not _refresh_stop.is_set():
        try:
            refresh_cookie_file(target_path=configured_cookie_target())
        except Exception as exc:
            logger.warning(f"Scheduled cookie refresh failed: {exc}")
        if _refresh_stop.wait(interval_seconds):
            break


def start_periodic_cookie_refresh() -> bool:
    """Start one daemon refresher when an endpoint and interval are configured."""

    global _refresh_thread
    if not os.getenv(COOKIE_REFRESH_URL_ENV, "").strip():
        return False
    interval = float(os.getenv(COOKIE_REFRESH_INTERVAL_SECONDS_ENV, "3600"))
    if interval <= 0:
        return False
    with _refresh_thread_lock:
        if _refresh_thread is not None and _refresh_thread.is_alive():
            return False
        _refresh_stop.clear()
        _refresh_thread = threading.Thread(
            target=_periodic_refresh_loop,
            args=(max(1.0, interval),),
            name="chronoseek-cookie-refresh",
            daemon=True,
        )
        _refresh_thread.start()
    logger.info(f"Scheduled cookie refresh every {interval:g} seconds.")
    return True


def stop_periodic_cookie_refresh() -> None:
    global _refresh_thread
    _refresh_stop.set()
    with _refresh_thread_lock:
        thread = _refresh_thread
        _refresh_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
