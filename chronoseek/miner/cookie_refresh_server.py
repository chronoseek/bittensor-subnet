"""Miner-operated endpoint serving fresh cookies to bound Chutes runtimes."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response, status

from chronoseek.miner.cookie_refresh import sign_refresh_request


REGISTRY_PATH_ENV = "COOKIE_REFRESH_REGISTRY_PATH"
MAX_CLOCK_SKEW_SECONDS_ENV = "COOKIE_REFRESH_MAX_CLOCK_SKEW_SECONDS"
_used_nonces: dict[tuple[str, str], float] = {}
_nonce_lock = threading.Lock()


def _build_server_logger() -> logging.Logger:
    logger = logging.getLogger("chronoseek.cookie_refresh_server")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | cookie-refresh | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    logger.disabled = False
    return logger


server_logger = _build_server_logger()

app = FastAPI(title="ChronoSeek Miner Cookie Refresh", docs_url=None, redoc_url=None)


def _registry_path() -> Path:
    value = os.getenv(REGISTRY_PATH_ENV, "").strip()
    if not value:
        raise RuntimeError(f"{REGISTRY_PATH_ENV} is required.")
    return Path(value).expanduser().resolve()


def load_binding(binding_id: str) -> tuple[str, Path]:
    """Reload registry on every call so rotations require no process restart."""

    payload = json.loads(_registry_path().read_text(encoding="utf-8"))
    bindings = payload.get("bindings", {}) if isinstance(payload, dict) else {}
    binding = bindings.get(binding_id) if isinstance(bindings, dict) else None
    if not isinstance(binding, dict):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown binding.")
    secret = str(binding.get("secret") or "")
    cookie_path = Path(str(binding.get("cookie_file") or "")).expanduser().resolve()
    if len(secret) < 32:
        raise HTTPException(status_code=500, detail="Binding secret is not configured securely.")
    return secret, cookie_path


def consume_nonce(binding_id: str, nonce: str, now: float, ttl: float) -> None:
    with _nonce_lock:
        expired = [key for key, expiry in _used_nonces.items() if expiry <= now]
        for key in expired:
            _used_nonces.pop(key, None)
        key = (binding_id, nonce)
        if key in _used_nonces:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Nonce replayed.")
        _used_nonces[key] = now + ttl


@app.get("/health")
def health() -> dict:
    try:
        path = _registry_path()
        return {"ok": path.is_file(), "service": "chronoseek-cookie-refresh"}
    except Exception:
        return {"ok": False, "service": "chronoseek-cookie-refresh"}


@app.post("/v1/cookies/refresh")
def refresh_cookies(
    request: Request,
    x_chronoseek_binding: str = Header(alias="X-Chronoseek-Binding"),
    x_chronoseek_timestamp: str = Header(alias="X-Chronoseek-Timestamp"),
    x_chronoseek_nonce: str = Header(alias="X-Chronoseek-Nonce"),
    x_chronoseek_signature: str = Header(alias="X-Chronoseek-Signature"),
) -> Response:
    client_host = request.client.host if request.client else "unknown"
    server_logger.info(
        "Received refresh request binding=%s client=%s",
        x_chronoseek_binding,
        client_host,
    )
    try:
        secret, cookie_path = load_binding(x_chronoseek_binding)
    except HTTPException as exc:
        server_logger.error(
            "Rejected refresh request binding=%s client=%s status=%s detail=%s",
            x_chronoseek_binding,
            client_host,
            exc.status_code,
            exc.detail,
        )
        raise
    except Exception:
        server_logger.exception(
            "Failed to load refresh binding=%s client=%s",
            x_chronoseek_binding,
            client_host,
        )
        raise HTTPException(
            status_code=500,
            detail="Cookie refresh server configuration error.",
        ) from None
    try:
        request_time = int(x_chronoseek_timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid timestamp.") from exc
    max_skew = max(5, int(os.getenv(MAX_CLOCK_SKEW_SECONDS_ENV, "30")))
    now = time.time()
    if abs(now - request_time) > max_skew:
        raise HTTPException(status_code=401, detail="Expired request.")
    if len(x_chronoseek_nonce) != 32:
        raise HTTPException(status_code=401, detail="Invalid nonce.")

    expected = sign_refresh_request(
        binding_id=x_chronoseek_binding,
        timestamp=x_chronoseek_timestamp,
        nonce=x_chronoseek_nonce,
        secret=secret,
    )
    if not hmac.compare_digest(expected, x_chronoseek_signature):
        server_logger.warning(
            "Rejected cookie refresh: invalid signature binding=%s client=%s",
            x_chronoseek_binding,
            client_host,
        )
        raise HTTPException(status_code=403, detail="Invalid signature.")

    consume_nonce(x_chronoseek_binding, x_chronoseek_nonce, now, max_skew * 2)

    try:
        cookie_bytes = cookie_path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Cookie file is unavailable.") from exc
    if not cookie_bytes.startswith(b"# Netscape HTTP Cookie File"):
        raise HTTPException(status_code=503, detail="Cookie file is invalid.")

    cookie_version = hashlib.sha256(cookie_bytes).hexdigest()[:16]
    server_logger.info(
        "Served fresh cookies binding=%s client=%s version=%s bytes=%d",
        x_chronoseek_binding,
        client_host,
        cookie_version,
        len(cookie_bytes),
    )
    return Response(
        content=cookie_bytes,
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-Chronoseek-Cookie-Version": cookie_version,
        },
    )
