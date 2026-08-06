import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from chronoseek.protocol_models import VideoSearchRequest
from chronoseek.validator.forward import query_miner, run_step
from chronoseek.chain.submissions import (
    DuplicateMinerSubmissionError,
    MinerSubmission,
    PERMANENT_SUBMISSION_ERROR,
    commit_miner_submission,
    load_chain_submission_snapshot,
    load_chain_submissions,
)
from chronoseek.chutes.runtime import (
    ChutesRuntimeEndpoint,
    build_evaluation_endpoints,
    build_submission_endpoint_map,
    check_runtime_health,
    chutes_auth_headers_from_env,
    resolve_submission_endpoint,
)
from chronoseek.utils import (
    build_endpoint_map_with_axon_fallback,
    normalize_endpoint_scheme,
    resolve_axon_endpoint,
)


class DummyMetagraph:
    def __init__(self):
        self.uids = [0, 1]
        self.hotkeys = ["hk-0", "hk-1"]


class DummyMetagraphWithAxons(DummyMetagraph):
    """Adds Bittensor 11's per-neuron `axon` field (`ip:port` str or None)."""

    def __init__(self, *, uid_axons: dict[int, str | None], hotkeys=None):
        super().__init__()
        if hotkeys is not None:
            self.hotkeys = hotkeys
        self.neurons = [
            SimpleNamespace(axon=uid_axons.get(uid))
            for uid in range(len(self.hotkeys))
        ]


def revealed_commitment(*entries):
    return SimpleNamespace(revealed=list(entries), encrypted=True)


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


def test_chutes_auth_headers_use_bearer_token(monkeypatch):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_test")

    headers = chutes_auth_headers_from_env()

    assert headers["Authorization"] == "Bearer cpk_test"
    assert headers["X-Chutes-Version"]


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


def test_resolve_axon_endpoint_returns_none_when_unserved():
    assert resolve_axon_endpoint(None) is None
    assert resolve_axon_endpoint(SimpleNamespace(axon=None)) is None
    assert resolve_axon_endpoint(SimpleNamespace(axon="")) is None


def test_resolve_axon_endpoint_returns_ip_port_when_served():
    assert resolve_axon_endpoint(SimpleNamespace(axon="1.2.3.4:9000")) == "http://1.2.3.4:9000"


def test_resolve_axon_endpoint_does_not_double_prefix_existing_scheme():
    assert (
        resolve_axon_endpoint(SimpleNamespace(axon="https://1.2.3.4:9000"))
        == "https://1.2.3.4:9000"
    )


def test_endpoint_map_with_axon_fallback_prefers_chutes_when_both_exist():
    metagraph = DummyMetagraphWithAxons(uid_axons={1: "9.9.9.9:9000"})
    endpoint_map, sources = build_endpoint_map_with_axon_fallback(
        metagraph=metagraph,
        submissions_by_hotkey={
            "hk-1": MinerSubmission(hotkey="hk-1", endpoint="https://runtime.example.com"),
        },
        chutes_base_domain="chutes.ai",
    )

    assert endpoint_map == {1: "https://runtime.example.com"}
    assert sources == {1: "chutes"}


def test_endpoint_map_with_axon_fallback_uses_axon_when_no_commitment():
    metagraph = DummyMetagraphWithAxons(uid_axons={1: "9.9.9.9:9000"})
    endpoint_map, sources = build_endpoint_map_with_axon_fallback(
        metagraph=metagraph,
        submissions_by_hotkey={},
        chutes_base_domain="chutes.ai",
    )

    assert endpoint_map == {1: "http://9.9.9.9:9000"}
    assert sources == {1: "axon"}


def test_endpoint_map_with_axon_fallback_treats_chute_id_only_as_no_endpoint():
    """A chute_id-only submission doesn't resolve to a URL, so it must still
    fall back to axon exactly as if there were no submission at all."""
    metagraph = DummyMetagraphWithAxons(uid_axons={1: "9.9.9.9:9000"})
    endpoint_map, sources = build_endpoint_map_with_axon_fallback(
        metagraph=metagraph,
        submissions_by_hotkey={
            "hk-1": MinerSubmission(hotkey="hk-1", chute_id="unroutable-id-only"),
        },
        chutes_base_domain="chutes.ai",
    )

    assert endpoint_map == {1: "http://9.9.9.9:9000"}
    assert sources == {1: "axon"}


def test_endpoint_map_with_axon_fallback_skips_disqualified_uids():
    metagraph = DummyMetagraphWithAxons(uid_axons={1: "9.9.9.9:9000"})
    endpoint_map, sources = build_endpoint_map_with_axon_fallback(
        metagraph=metagraph,
        submissions_by_hotkey={},
        chutes_base_domain="chutes.ai",
        disqualified_uids={1},
    )

    assert endpoint_map == {}
    assert sources == {}


def test_endpoint_map_with_axon_fallback_no_endpoint_when_neither_exists():
    metagraph = DummyMetagraphWithAxons(uid_axons={})
    endpoint_map, sources = build_endpoint_map_with_axon_fallback(
        metagraph=metagraph,
        submissions_by_hotkey={},
        chutes_base_domain="chutes.ai",
    )

    assert endpoint_map == {}
    assert sources == {}


class TestAsyncSubmissionRouting(unittest.IsolatedAsyncioTestCase):
    async def test_load_chain_submissions_disqualifies_duplicate_commit_hotkey(self):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-1": revealed_commitment(
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

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
        )

        assert "hk-1" not in snapshot.submissions
        assert snapshot.duplicate_hotkeys == {"hk-1"}

    async def test_load_chain_submissions_uses_latest_when_enforcement_is_disabled(
        self,
    ):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-1": revealed_commitment(
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

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
            enforce_one_hotkey_one_submission=False,
        )

        assert snapshot.duplicate_hotkeys == set()
        assert snapshot.submissions["hk-1"].chute_slug == "new-runtime"
        assert snapshot.submissions["hk-1"].created_at_block == 20

    async def test_load_chain_submissions_disqualifies_retry_after_invalid_first_commit(self):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-1": revealed_commitment(
                        (10, "not-json"),
                        (
                            20,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "retry-runtime",
                                }
                            ),
                        ),
                    )
                }

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
        )

        assert "hk-1" not in snapshot.submissions
        assert snapshot.duplicate_hotkeys == {"hk-1"}

    async def test_load_chain_submissions_disqualifies_later_revealed_cloned_slug(
        self,
    ):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-0": revealed_commitment(
                        (
                            10,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                    "hk-1": revealed_commitment(
                        (
                            20,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                }

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
        )

        assert "hk-0" in snapshot.submissions
        assert "hk-1" not in snapshot.submissions
        assert snapshot.duplicate_hotkeys == {"hk-1"}

    async def test_load_chain_submissions_disqualifies_all_but_first_in_three_way_clone(
        self,
    ):
        class ThreeHotkeyMetagraph:
            def __init__(self):
                self.uids = [0, 1, 2]
                self.hotkeys = ["hk-0", "hk-1", "hk-2"]

        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-0": revealed_commitment(
                        (
                            30,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                    "hk-1": revealed_commitment(
                        (
                            10,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                    "hk-2": revealed_commitment(
                        (
                            20,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                }

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=ThreeHotkeyMetagraph(),
        )

        # hk-1 revealed first (block 10) - it keeps the slug. hk-2 (block 20)
        # and hk-0 (block 30) both lose it, regardless of registration order.
        assert set(snapshot.submissions) == {"hk-1"}
        assert snapshot.duplicate_hotkeys == {"hk-0", "hk-2"}

    async def test_load_chain_submissions_keeps_hotkey_with_unique_slug(self):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-0": revealed_commitment(
                        (
                            10,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "shared-slug",
                                }
                            ),
                        )
                    ),
                    "hk-1": revealed_commitment(
                        (
                            20,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "unique-slug",
                                }
                            ),
                        )
                    ),
                }

        snapshot = await load_chain_submission_snapshot(
            FakeSubtensor(),
            netuid=1,
            metagraph=DummyMetagraph(),
        )

        assert set(snapshot.submissions) == {"hk-0", "hk-1"}
        assert snapshot.duplicate_hotkeys == set()

    async def test_load_chain_submissions_keeps_single_valid_commit(self):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {
                    "hk-1": revealed_commitment(
                        (
                            10,
                            json.dumps(
                                {
                                    "runtime": "chutes",
                                    "protocol": "chronoseek-runtime-v2",
                                    "chute_slug": "only-runtime",
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

        assert submissions["hk-1"].chute_slug == "only-runtime"
        assert submissions["hk-1"].created_at_block == 10

    async def test_commit_miner_submission_rejects_second_commit_for_hotkey(self):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)

            def commitments(self, netuid):
                assert netuid == 1
                return {"hk-1": revealed_commitment((10, "{}"))}

            def submit_call(self, *args, **kwargs):
                raise AssertionError("duplicate submission must not be committed")

        wallet = MagicMock()
        wallet.hotkey.ss58_address = "hk-1"

        with self.assertRaises(DuplicateMinerSubmissionError) as exc:
            await commit_miner_submission(
                subtensor=FakeSubtensor(),
                wallet=wallet,
                netuid=1,
                submission=MinerSubmission(
                    hotkey="hk-1",
                    endpoint="https://runtime.example.com",
                ),
                blocks_until_reveal=1,
            )

        assert str(exc.exception) == PERMANENT_SUBMISSION_ERROR

    @patch(
        "chronoseek.chain.submissions.get_encrypted_commitment",
        return_value=(b"encrypted", 12345),
    )
    async def test_commit_miner_submission_skips_duplicate_check_when_disabled(
        self,
        mock_get_encrypted_commitment,
    ):
        class FakeSubtensor:
            def __init__(self):
                self.subnets = SimpleNamespace(commitments=self.commitments)
                self.submitted = False

            def commitments(self, netuid):
                raise AssertionError("duplicate history must not be read")

            def submit_call(self, *args, **kwargs):
                self.submitted = True
                return True

        subtensor = FakeSubtensor()
        wallet = MagicMock()
        wallet.hotkey.ss58_address = "hk-1"

        success = await commit_miner_submission(
            subtensor=subtensor,
            wallet=wallet,
            netuid=1,
            submission=MinerSubmission(
                hotkey="hk-1",
                endpoint="https://runtime.example.com",
            ),
            blocks_until_reveal=1,
            enforce_one_hotkey_one_submission=False,
        )

        assert success is True
        assert subtensor.submitted is True
        mock_get_encrypted_commitment.assert_called_once()

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


def test_normalize_endpoint_scheme_prefixes_bare_host():
    assert normalize_endpoint_scheme("9.9.9.9:9000") == "http://9.9.9.9:9000"


def test_normalize_endpoint_scheme_leaves_existing_scheme_alone():
    assert (
        normalize_endpoint_scheme("https://runtime.example.com")
        == "https://runtime.example.com"
    )


class TestCheckRuntimeHealthRealTransport(unittest.IsolatedAsyncioTestCase):
    """Regression coverage for PR #33 review feedback: the previous test
    suite mocked httpx.AsyncClient entirely, so a bare ip:port endpoint
    never actually went through httpx's own URL parsing/validation and the
    missing-scheme bug (httpx.UnsupportedProtocol) went uncaught. These use
    a real httpx.AsyncClient with only its transport swapped out, so the
    real URL-parsing code path runs.

    check_runtime_health itself does not normalize its endpoint argument -
    that contract hasn't changed for the Chutes path (resolve_submission_
    endpoint has always returned a full URL) and isn't changing for the
    axon path either: normalization happens once, upstream, in
    chronoseek/utils.py's resolve_axon_endpoint/normalize_endpoint_scheme,
    before an endpoint ever reaches endpoint_map. These tests cover both
    ends of that contract with a real transport.
    """

    async def test_axon_endpoint_is_pre_normalized_before_reaching_health_check(self):
        """End-to-end: an axon's bare ip:port, once resolved via
        resolve_axon_endpoint, must already be a full URL that
        check_runtime_health can use as-is against a real client."""
        neuron = SimpleNamespace(axon="9.9.9.9:9000")
        resolved_endpoint = resolve_axon_endpoint(neuron)
        self.assertEqual(resolved_endpoint, "http://9.9.9.9:9000")

        requested_urls = []

        def handler(request):
            requested_urls.append(str(request.url))
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            is_healthy = await check_runtime_health(
                client=client,
                uid=7,
                endpoint=resolved_endpoint,
                timeout_seconds=5,
            )

        self.assertTrue(is_healthy)
        self.assertEqual(requested_urls, ["http://9.9.9.9:9000/health"])

    async def test_check_runtime_health_still_works_for_full_chutes_url(self):
        def handler(request):
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            is_healthy = await check_runtime_health(
                client=client,
                uid=1,
                endpoint="https://runtime.example.com",
                timeout_seconds=5,
            )

        self.assertTrue(is_healthy)

    async def test_check_runtime_health_does_not_normalize_a_bare_endpoint_itself(self):
        """Documents the actual contract: check_runtime_health assumes a
        pre-normalized endpoint. A bare ip:port passed directly (bypassing
        resolve_axon_endpoint) fails - which is exactly why normalization
        must happen upstream, not defensively here."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True})
        )) as client:
            is_healthy = await check_runtime_health(
                client=client,
                uid=7,
                endpoint="9.9.9.9:9000",
                timeout_seconds=5,
            )

        self.assertFalse(is_healthy)
