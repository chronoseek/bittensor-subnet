import threading
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from chronoseek.validator.gateway import ValidatorGatewayRuntime, create_validator_gateway
from chronoseek.validator.forward import MinerQueryFailure, MinerQueryResult
from chronoseek.protocol_models import VideoSearchResponse


class DummyAxon:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port


class DummyMetagraph:
    def __init__(self):
        self.uids = [0, 1]
        self.axons = [DummyAxon("1.1.1.1", 8000), DummyAxon("2.2.2.2", 8000)]


def test_gateway_health_endpoint():
    runtime = ValidatorGatewayRuntime(
        wallet=None,
        metagraph=DummyMetagraph(),
        scores=np.zeros(2),
        score_lock=threading.Lock(),
        max_miners_per_request=2,
        miner_request_timeout_seconds=60.0,
    )

    client = TestClient(create_validator_gateway(runtime))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_gateway_capabilities_endpoint():
    runtime = ValidatorGatewayRuntime(
        wallet=None,
        metagraph=DummyMetagraph(),
        scores=np.zeros(2),
        score_lock=threading.Lock(),
        max_miners_per_request=2,
        miner_request_timeout_seconds=60.0,
    )

    client = TestClient(create_validator_gateway(runtime))
    response = client.get("/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "validator-gateway",
        "protocol_versions": ["2026-03-01"],
    }


@patch("chronoseek.validator.gateway.query_miner")
def test_gateway_search_returns_protocol_response(mock_query_miner):
    runtime = ValidatorGatewayRuntime(
        wallet=None,
        metagraph=DummyMetagraph(),
        scores=np.array([0.9, 0.1]),
        score_lock=threading.Lock(),
        max_miners_per_request=2,
        miner_request_timeout_seconds=60.0,
    )
    mock_query_miner.side_effect = [
        MinerQueryResult(
            response=VideoSearchResponse(
                request_id="req-1",
                status="completed",
                results=[
                    {"start": 1.0, "end": 3.5, "confidence": 0.8},
                    {"start": 5.0, "end": 8.0, "confidence": 0.6},
                ],
            ),
            latency=0.2,
        ),
        MinerQueryResult(
            response=VideoSearchResponse(
                request_id="req-1",
                status="completed",
                results=[
                    {"start": 2.0, "end": 4.0, "confidence": 0.95},
                ],
            ),
            latency=0.3,
        ),
    ]

    client = TestClient(create_validator_gateway(runtime))
    response = client.post(
        "/search",
        json={
            "protocol_version": "2026-03-01",
            "request_id": "req-1",
            "query": "people fighting",
            "top_k": 5,
            "video": {"url": "https://example.com/video.mp4"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert isinstance(body["results"], list)
    assert body["results"][0]["start"] == 2.0
    assert body["results"][0]["confidence"] == 0.95
    assert body["results"][1]["start"] == 1.0
    assert body["miner_metadata"]["source"] == "validator-gateway"
    assert body["miner_metadata"]["selected_uids"] == [0, 1]


@patch("chronoseek.validator.gateway.query_miner")
def test_gateway_search_returns_structured_timeout_error(mock_query_miner):
    runtime = ValidatorGatewayRuntime(
        wallet=None,
        metagraph=DummyMetagraph(),
        scores=np.array([0.9, 0.1]),
        score_lock=threading.Lock(),
        max_miners_per_request=2,
        miner_request_timeout_seconds=60.0,
    )
    mock_query_miner.return_value = MinerQueryResult(
        response=VideoSearchResponse(results=[]),
        latency=0.0,
        failure=MinerQueryFailure(kind="timeout", message="request timed out"),
    )

    client = TestClient(create_validator_gateway(runtime))
    response = client.post(
        "/search",
        json={
            "protocol_version": "2026-03-01",
            "request_id": "req-2",
            "query": "people fighting",
            "top_k": 5,
            "video": {"url": "https://example.com/video.mp4"},
        },
    )

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "TIMEOUT"
