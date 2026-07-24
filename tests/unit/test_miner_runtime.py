import asyncio
import os
from unittest.mock import MagicMock, patch

import httpx

import chronoseek.miner.runtime as runtime
from chronoseek.epistula import verify_signature
from chronoseek.miner.logic import SearchPipelineError
from chronoseek.protocol_models import VideoSearchRequest, VideoSearchResult


def test_runtime_config_defaults_to_finney_netuid_20():
    with patch.dict(os.environ, {}, clear=True):
        config = runtime.load_runtime_config()

    assert config.network == "finney"
    assert config.netuid == 20


def test_runtime_config_accepts_explicit_chain_overrides():
    with patch.dict(
        os.environ,
        {"NETWORK": "test", "NETUID": "298"},
        clear=True,
    ):
        config = runtime.load_runtime_config()

    assert config.network == "test"
    assert config.netuid == 298


def test_execute_search_returns_json_serializable_dict():
    previous_miner_logic = runtime.miner_logic
    previous_validator_auth = runtime.validator_auth
    runtime.miner_logic = MagicMock()
    runtime.miner_logic.search.return_value = [
        VideoSearchResult(start=1.0, end=2.0, confidence=0.9)
    ]
    runtime.validator_auth = MagicMock()

    try:
        response = runtime.execute_search(
            VideoSearchRequest(
                request_id="req-direct",
                video={"url": "https://example.com/video.mp4"},
                query="a person waves",
                top_k=1,
            ),
            enforce_validator_auth=False,
        )

        assert isinstance(response, dict)
        assert response == {
            "protocol_version": "2026-04-10",
            "request_id": "req-direct",
            "status": "completed",
            "results": [
                {"start": 1.0, "end": 2.0, "confidence": 0.9},
            ],
            "miner_metadata": None,
        }
    finally:
        runtime.miner_logic = previous_miner_logic
        runtime.validator_auth = previous_validator_auth


def test_chutes_runtime_exposes_health_and_search_contract():
    miner_logic = MagicMock()
    miner_logic.search.return_value = [
        VideoSearchResult(start=1.0, end=2.0, confidence=0.9)
    ]

    def initialize_runtime_stub():
        runtime.miner_logic = miner_logic
        runtime.validator_auth = MagicMock()
        runtime.startup_error = None

    async def verify_signature_stub():
        return "validator-hotkey"

    runtime.app.dependency_overrides[verify_signature] = verify_signature_stub
    try:
        with patch.object(runtime, "initialize_runtime", initialize_runtime_stub), patch.object(
            runtime,
            "authorize_hotkey",
            return_value=(True, {"caller_stake": 1.0, "minimum_validator_stake": 0.0}),
        ):
            async def exercise_runtime():
                transport = httpx.ASGITransport(app=runtime.app)
                async with runtime.lifespan(runtime.app), httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    health = await client.get("/health")
                    response = await client.post(
                        "/search",
                        json={
                            "protocol_version": "2026-04-10",
                            "request_id": "req-1",
                            "video": {"url": "https://example.com/video.mp4"},
                            "query": "a person waves",
                            "top_k": 1,
                        },
                    )
                    return health, response

            health, response = asyncio.run(exercise_runtime())
            assert health.status_code == 200
            assert health.json()["ok"] is True

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


def test_chutes_runtime_returns_structured_video_fetch_error():
    miner_logic = MagicMock()
    miner_logic.search.side_effect = SearchPipelineError(
        "VIDEO_FETCH_FAILED",
        "The video URL could not be fetched by supported download backends.",
        {
            "video_url": "https://s3.hippius.com/chronoseek/task-clips/task.mp4",
            "download_attempts": [
                {"backend": "hippius-s3", "message": "403 Forbidden"},
                {"backend": "http", "message": "403 Forbidden"},
            ],
        },
    )

    def initialize_runtime_stub():
        runtime.miner_logic = miner_logic
        runtime.validator_auth = MagicMock()
        runtime.startup_error = None

    async def verify_signature_stub():
        return "validator-hotkey"

    runtime.app.dependency_overrides[verify_signature] = verify_signature_stub
    try:
        with patch.object(runtime, "initialize_runtime", initialize_runtime_stub), patch.object(
            runtime,
            "authorize_hotkey",
            return_value=(True, {"caller_stake": 1.0, "minimum_validator_stake": 0.0}),
        ):
            async def exercise_runtime():
                transport = httpx.ASGITransport(app=runtime.app)
                async with runtime.lifespan(runtime.app), httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/search",
                        json={
                            "protocol_version": "2026-04-10",
                            "request_id": "req-fetch-fail",
                            "video": {
                                "url": "https://s3.hippius.com/chronoseek/task-clips/task.mp4"
                            },
                            "query": "a person waves",
                            "top_k": 1,
                        },
                    )

            response = asyncio.run(exercise_runtime())
            assert response.status_code == 502
            body = response.json()
            assert body["error"]["code"] == "VIDEO_FETCH_FAILED"
            assert body["error"]["details"]["request_id"] == "req-fetch-fail"
            assert body["error"]["details"]["download_attempts"] == [
                {"backend": "hippius-s3", "message": "403 Forbidden"},
                {"backend": "http", "message": "403 Forbidden"},
            ]
    finally:
        runtime.app.dependency_overrides.clear()
        runtime.miner_logic = None
        runtime.validator_auth = None
        runtime.startup_error = None
