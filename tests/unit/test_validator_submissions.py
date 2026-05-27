import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from chronoseek.protocol_models import VideoSearchRequest
from chronoseek.validator.forward import query_miner, run_step
from chronoseek.chain.submissions import (
    MinerSubmission,
    load_chain_submissions,
)
from chronoseek.chutes.runtime import (
    ChutesRuntimeEndpoint,
    build_evaluation_endpoints,
    build_submission_endpoint_map,
    resolve_submission_endpoint,
)


class DummyMetagraph:
    def __init__(self):
        self.uids = [0, 1]
        self.hotkeys = ["hk-0", "hk-1"]


def test_submission_endpoint_resolves_explicit_endpoint_before_slug():
    submission = MinerSubmission(
        hotkey="hk-1",
        endpoint="https://private-runtime.example.com",
        chute_slug="ignored-slug",
    )

    assert (
        resolve_submission_endpoint(submission, chutes_base_domain="chutes.ai")
        == "https://private-runtime.example.com"
    )


def test_submission_endpoint_resolves_chutes_slug():
    submission = MinerSubmission(hotkey="hk-1", chute_slug="chronoseek-runtime")

    assert (
        resolve_submission_endpoint(submission, chutes_base_domain="chutes.ai")
        == "https://chronoseek-runtime.chutes.ai"
    )


def test_build_evaluation_endpoints_uses_registered_submission():
    metagraph = DummyMetagraph()
    endpoints = build_evaluation_endpoints(
        metagraph=metagraph,
        candidate_uids=None,
        submissions_by_hotkey={
            "hk-1": MinerSubmission(
                hotkey="hk-1",
                endpoint="https://runtime.example.com",
            )
        },
        chutes_base_domain="chutes.ai",
    )

    assert [(endpoint.uid, endpoint.endpoint) for endpoint in endpoints] == [
        (1, "https://runtime.example.com")
    ]


def test_build_evaluation_endpoints_uses_submissions_without_endpoint_advertisements():
    endpoints = build_evaluation_endpoints(
        metagraph=DummyMetagraph(),
        candidate_uids=None,
        submissions_by_hotkey={
            "hk-1": MinerSubmission(
                hotkey="hk-1",
                endpoint="https://runtime.example.com",
            )
        },
        chutes_base_domain="chutes.ai",
    )

    assert [(endpoint.uid, endpoint.endpoint) for endpoint in endpoints] == [
        (1, "https://runtime.example.com")
    ]


def test_submission_endpoint_map_uses_registered_hotkeys_only():
    endpoint_map = build_submission_endpoint_map(
        metagraph=DummyMetagraph(),
        submissions_by_hotkey={
            "hk-1": MinerSubmission(
                hotkey="hk-1",
                endpoint="https://runtime.example.com",
            ),
            "not-in-metagraph": MinerSubmission(
                hotkey="not-in-metagraph",
                endpoint="https://ignored.example.com",
            ),
        },
        chutes_base_domain="chutes.ai",
    )

    assert endpoint_map == {1: "https://runtime.example.com"}


class TestAsyncSubmissionRouting(unittest.IsolatedAsyncioTestCase):
    async def test_load_chain_submissions_uses_latest_commit_by_hotkey(self):
        class FakeSubtensor:
            def get_all_revealed_commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-1": (
                        (
                            10,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "old-runtime",
                                }
                            ),
                        ),
                        (
                            20,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "new-runtime",
                                }
                            ),
                        ),
                    )
                }

        submissions = await load_chain_submissions(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
        )

        assert submissions["hk-1"].chute_slug == "new-runtime"
        assert submissions["hk-1"].created_at_block == 20

    @patch("chronoseek.validator.forward.generate_header")
    async def test_query_miner_adds_provider_headers(self, mock_generate_header):
        wallet = MagicMock()
        wallet.hotkey = MagicMock()
        mock_generate_header.return_value = {"X-Epistula-Hotkey": "validator-hotkey"}

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "protocol_version": "2026-04-10",
            "request_id": "req-1",
            "status": "completed",
            "results": [],
        }
        client = AsyncMock()
        client.post.return_value = response

        request = VideoSearchRequest(
            request_id="req-1",
            video={"url": "https://example.com/video.mp4"},
            query="a person speaks",
        )

        await query_miner(
            client=client,
            uid=1,
            hotkey="hk-1",
            endpoint="https://runtime.example.com",
            request=request,
            wallet=wallet,
            extra_headers={"Authorization": "Bearer secret"},
        )

        assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
        assert (
            client.post.call_args.kwargs["headers"]["X-Epistula-Hotkey"]
            == "validator-hotkey"
        )

    @patch("chronoseek.validator.forward.generate_header")
    async def test_run_step_routes_synthetic_eval_to_submission_endpoint(
        self,
        mock_generate_header,
    ):
        mock_generate_header.return_value = {"X-Epistula-Hotkey": "validator-hotkey"}

        task_gen = MagicMock()
        task_gen.generate_task.return_value = (
            "https://example.com/video.mp4",
            "a person speaks",
            [(1.0, 3.0)],
        )

        wallet = MagicMock()
        wallet.hotkey = MagicMock()
        client = AsyncMock()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "protocol_version": "2026-04-10",
            "status": "completed",
            "results": [{"start": 1.0, "end": 3.0, "confidence": 0.9}],
        }
        client.post.return_value = response

        scores = await run_step(
            task_gen=task_gen,
            metagraph=DummyMetagraph(),
            wallet=wallet,
            client=client,
            miner_timeout_seconds=10,
            miner_endpoints=[
                ChutesRuntimeEndpoint(
                    uid=1,
                    hotkey="hk-1",
                    endpoint="https://runtime.example.com",
                )
            ],
            provider_headers={"Authorization": "Bearer secret"},
        )

        assert scores == [(1, 1.0)]
        assert client.post.call_args.args[0] == "https://runtime.example.com/search"
        assert (
            client.post.call_args.kwargs["headers"]["Authorization"]
            == "Bearer secret"
        )
