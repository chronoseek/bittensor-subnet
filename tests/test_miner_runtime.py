from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import chronoseek.miner.runtime as runtime
from chronoseek.epistula import verify_signature
from chronoseek.protocol_models import VideoSearchResult


def test_chutes_runtime_exposes_health_and_search_contract():
    miner_logic = MagicMock()
    miner_logic.search.return_value = [
        VideoSearchResult(start=1.0, end=2.0, confidence=0.9)
    ]

    def initialize_runtime_stub():
        runtime.miner_logic = miner_logic
        runtime.validator_auth = MagicMock()
        runtime.startup_error = None

    runtime.app.dependency_overrides[verify_signature] = lambda: "validator-hotkey"
    try:
        with patch.object(runtime, "initialize_runtime", initialize_runtime_stub), patch.object(
            runtime,
            "authorize_hotkey",
            return_value=(True, {"caller_stake": 1.0, "minimum_validator_stake": 0.0}),
        ):
            with TestClient(runtime.app) as client:
                health = client.get("/health")
                assert health.status_code == 200
                assert health.json()["ok"] is True

                response = client.post(
                    "/search",
                    json={
                        "protocol_version": "2026-04-10",
                        "request_id": "req-1",
                        "video": {"url": "https://example.com/video.mp4"},
                        "query": "a person waves",
                        "top_k": 1,
                    },
                )

                assert response.status_code == 200
                body = response.json()
                assert body["status"] == "completed"
                assert body["results"] == [
                    {"start": 1.0, "end": 2.0, "confidence": 0.9}
                ]
                miner_logic.search.assert_called_once_with(
                    "https://example.com/video.mp4",
                    "a person waves",
                    top_k=1,
                )
    finally:
        runtime.app.dependency_overrides.clear()
        runtime.miner_logic = None
        runtime.validator_auth = None
        runtime.startup_error = None
