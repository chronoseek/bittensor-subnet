import time
import hashlib
import json
import bittensor as bt
from typing import Optional
from fastapi import Request, HTTPException


def generate_header(hotkey: bt.Keypair, body: dict) -> dict:
    """
    Generate Epistula headers for a request.
    """
    timestamp = str(int(time.time() * 1000))
    body_bytes = json.dumps(body, sort_keys=True).encode("utf-8")
    body_hash = hashlib.sha256(body_bytes).hexdigest()

    message = f"{timestamp}.{body_hash}"
    signature = f"0x{hotkey.sign(message).hex()}"

    return {
        "X-Epistula-Timestamp": timestamp,
        "X-Epistula-Signature": signature,
        "X-Epistula-Hotkey": hotkey.ss58_address,
        "Content-Type": "application/json",
    }


async def verify_signature(request: Request) -> str:
    """
    Verify Epistula signature from a FastAPI request.
    Returns the signer's hotkey address if valid.
    Raises HTTPException if invalid.
    """
    headers = request.headers
    timestamp = headers.get("X-Epistula-Timestamp")
    signature = headers.get("X-Epistula-Signature")
    sender_hotkey = headers.get("X-Epistula-Hotkey")

    if not all([timestamp, signature, sender_hotkey]):
        raise HTTPException(status_code=401, detail="Missing Epistula headers")

    # 1. Verify timestamp (prevent replay attacks, allow 60s drift)
    try:
        ts = int(timestamp)
        now = int(time.time() * 1000)
        if abs(now - ts) > 60000:
            raise HTTPException(status_code=401, detail="Request expired")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    # 2. Reconstruct message
    body = await request.json()
    body_bytes = json.dumps(body, sort_keys=True).encode("utf-8")
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    message = f"{timestamp}.{body_hash}"

    # 3. Verify signature
    try:
        if signature.startswith("0x"):
            signature = signature[2:]

        signature_bytes = bytes.fromhex(signature)

        # Verify using bittensor Keypair
        if not bt.Keypair(ss58_address=sender_hotkey).verify(message, signature_bytes):
            raise HTTPException(status_code=401, detail="Invalid signature")

    except Exception as e:
        raise HTTPException(
            status_code=401, detail=f"Signature verification failed: {str(e)}"
        )

    return sender_hotkey
