import unittest

from bittensor.sp_core import Keypair
from fastapi import HTTPException

from chronoseek.epistula import generate_header, verify_signature


class FakeRequest:
    def __init__(self, headers: dict, body: dict):
        self.headers = headers
        self._body = body

    async def json(self):
        return self._body


def _keypair() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


class TestVerifySignature(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_a_real_generated_header(self):
        """Regression test: Keypair.verify() requires the message as bytes,
        not str. Passing the str message directly always raised
        "argument 'message': 'str' object cannot be converted to 'PyBytes'",
        which surfaced as a 401 on every real signed request (e.g. the axon
        serving track's first live validator query)."""
        hotkey = _keypair()
        body = {"request_id": "test-1", "query": "test"}
        headers = generate_header(hotkey, body)

        caller_hotkey = await verify_signature(FakeRequest(headers, body))

        self.assertEqual(caller_hotkey, hotkey.ss58_address)

    async def test_rejects_tampered_body(self):
        hotkey = _keypair()
        body = {"request_id": "test-1", "query": "test"}
        headers = generate_header(hotkey, body)

        tampered_body = {"request_id": "test-1", "query": "tampered"}

        with self.assertRaises(HTTPException) as ctx:
            await verify_signature(FakeRequest(headers, tampered_body))

        self.assertEqual(ctx.exception.status_code, 401)

    async def test_rejects_missing_headers(self):
        with self.assertRaises(HTTPException) as ctx:
            await verify_signature(FakeRequest({}, {"request_id": "test-1"}))

        self.assertEqual(ctx.exception.status_code, 401)
