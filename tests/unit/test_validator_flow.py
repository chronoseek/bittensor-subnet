import unittest
import asyncio
import threading
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

# Adjust path if needed
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validator import (
    apply_disqualified_miner_penalty,
    build_emission_weights,
    refresh_responsive_miners_from_submissions,
    run_validator_loop,
    seed_scores_from_metagraph,
    top3_weight_shares,
)
from chronoseek.protocol_models import VideoSearchRequest
from chronoseek.validator.forward import query_miner
from chronoseek.validator.state import ValidatorRuntimeState
from chronoseek.chain.submissions import MinerSubmission


class TestValidatorFlow(unittest.IsolatedAsyncioTestCase):

    def test_build_emission_weights_reserves_burn_uid_and_pays_top3(self):
        # burn_uid (0) is excluded from ranking, leaving only 2 candidates -
        # rank3's 5% slot goes unpaid rather than falling back to anyone.
        scores = np.array([100.0, 2.0, 6.0])

        weights = build_emission_weights(scores, miner_emission_burn_percent=25.0)

        np.testing.assert_allclose(weights, np.array([0.25, 0.15, 0.5625]))

    def test_build_emission_weights_pays_top3_when_burn_is_zero(self):
        scores = np.array([100.0, 2.0, 6.0])

        weights = build_emission_weights(scores, miner_emission_burn_percent=0.0)

        np.testing.assert_allclose(weights, np.array([0.0, 0.20, 0.75]))

    def test_build_emission_weights_pays_full_top3_with_enough_candidates(self):
        scores = np.array([0.0, 6.0, 2.0, 9.0])

        weights = build_emission_weights(scores, miner_emission_burn_percent=0.0)

        np.testing.assert_allclose(weights, np.array([0.0, 0.20, 0.05, 0.75]))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_build_emission_weights_burns_all_when_no_rest_scores_exist(self):
        scores = np.array([0.0, 0.0, 0.0])

        weights = build_emission_weights(scores, miner_emission_burn_percent=30.0)

        np.testing.assert_allclose(weights, np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_top3_weight_shares_clean_ranking(self):
        scores = np.array([9.0, 2.0, 6.0, 1.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(shares, np.array([0.75, 0.05, 0.20, 0.0]))

    def test_top3_weight_shares_two_way_tie_for_first(self):
        scores = np.array([9.0, 9.0, 6.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(shares, np.array([0.475, 0.475, 0.05]))

    def test_top3_weight_shares_two_way_tie_for_second_and_third(self):
        scores = np.array([9.0, 6.0, 6.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(shares, np.array([0.75, 0.125, 0.125]))

    def test_top3_weight_shares_three_way_tie_for_first(self):
        scores = np.array([9.0, 9.0, 9.0, 1.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(
            shares, np.array([1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0])
        )

    def test_top3_weight_shares_five_way_tie_for_first(self):
        scores = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 1.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(
            shares, np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.0])
        )

    def test_top3_weight_shares_never_pays_zero_or_negative_score(self):
        scores = np.array([0.0, -1.0, 5.0])

        shares = top3_weight_shares(scores)

        np.testing.assert_allclose(shares, np.array([0.0, 0.0, 0.75]))

    def test_seed_scores_from_metagraph_uses_incentives(self):
        metagraph = MagicMock()
        metagraph.n = 3
        metagraph.I = [0.2, 0.3, 0.5]

        scores = seed_scores_from_metagraph(metagraph)

        np.testing.assert_allclose(scores, np.array([0.2, 0.3, 0.5]))

    def test_apply_disqualified_miner_penalty_zeros_duplicate_submitters(self):
        scores = np.array([0.4, 0.6, 0.9])

        penalized = apply_disqualified_miner_penalty(scores, {1})

        np.testing.assert_allclose(penalized, np.array([0.4, 0.0, 0.9]))

    @patch("chronoseek.validator.task_gen.ActivityNetTaskGenerator")
    @patch("chronoseek.validator.forward.run_step")
    async def test_weight_setting(self, mock_run_step, mock_task_gen):
        """
        Test that weights are set correctly when tempo is reached.
        """
        # Setup Mocks
        mock_subtensor = MagicMock()
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        stop_event = MagicMock()

        # Configure Metagraph (3 neurons)
        mock_metagraph.n = 3
        mock_metagraph.hotkeys = ["h1", "h2", "h3"]
        mock_metagraph.I = [0.2, 0.3, 0.5]

        # Configure Subtensor
        mock_subtensor.subnets.info.return_value.tempo = 5
        mock_subtensor.execute.return_value = True

        # Simulate block progression: 0 -> 1 -> ... -> 5 (trigger weights) -> 6 (stop)
        # We use a side_effect to increment blocks and eventually set stop_event
        block_counter = [1]

        def get_block_side_effect():
            current = block_counter[0]
            block_counter[0] += 1
            if current >= 6:
                stop_event.is_set.return_value = True  # Stop loop
            return current

        mock_subtensor.block.side_effect = get_block_side_effect
        # IMPORTANT: stop_event.is_set is called once per loop iteration
        # We need enough False returns to let the loop run until block 5
        stop_event.is_set.side_effect = [False] * 10 + [True]

        # Mock run_step to return some scores
        # Iteration 1: Miner 0 scores 1.0
        # Iteration 2: Miner 1 scores 0.5
        # ...
        async def run_step_side_effect(*args, **kwargs):
            miner_endpoints = kwargs["miner_endpoints"]
            self.assertEqual([endpoint.uid for endpoint in miner_endpoints], [0, 1, 2])
            self.assertEqual(
                miner_endpoints[0].endpoint,
                "https://runtime-0.example.com",
            )
            return [(0, 1.0), (1, 0.5)]

        mock_run_step.side_effect = run_step_side_effect

        # Mock sleep to be instant
        with patch("asyncio.sleep", new_callable=AsyncMock), patch(
            "validator.MinerSubmissionResolver"
        ) as mock_resolver_cls, patch(
            "validator.refresh_responsive_miners_from_submissions", new_callable=AsyncMock
        ) as mock_refresh_responsive:
            mock_resolver = MagicMock()
            mock_resolver.get_submissions = AsyncMock(
                return_value={
                    "h1": MinerSubmission(
                        hotkey="h1", endpoint="https://runtime-0.example.com"
                    ),
                    "h2": MinerSubmission(
                        hotkey="h2", endpoint="https://runtime-1.example.com"
                    ),
                    "h3": MinerSubmission(
                        hotkey="h3", endpoint="https://runtime-2.example.com"
                    ),
                }
            )
            mock_resolver_cls.return_value = mock_resolver

            async def refresh_side_effect(*args, **kwargs):
                refresh_runtime = kwargs["runtime"]
                with refresh_runtime.responsive_lock:
                    refresh_runtime.responsive_uids = {0, 1, 2}
                    refresh_runtime.miner_endpoints = {
                        0: "https://runtime-0.example.com",
                        1: "https://runtime-1.example.com",
                        2: "https://runtime-2.example.com",
                    }
                    refresh_runtime.responsive_initialized = True
                    refresh_runtime.responsive_last_refresh_at = 1.0
                return {0, 1, 2}

            mock_refresh_responsive.side_effect = refresh_side_effect
            runtime = ValidatorRuntimeState(
                wallet=mock_wallet,
                metagraph=mock_metagraph,
                scores=seed_scores_from_metagraph(mock_metagraph),
                score_lock=threading.Lock(),
            )
            await run_validator_loop(
                mock_subtensor,
                runtime,
                netuid=1,
                stop_event=stop_event,
                last_heartbeat=[0],
                config=MagicMock(
                    task_dataset_path="",
                    task_split="validation",
                    require_accessible_videos=False,
                    task_max_sampling_attempts=10,
                    miner_request_timeout_seconds=60.0,
                    miner_emission_burn_percent=0.0,
                    video_availability_cache_path="",
                    accessible_video_cache_path="",
                    inaccessible_video_cache_path="",
                    video_availability_cache_ttl_hours=24,
                    video_availability_timeout=5,
                    miner_submission_refresh_interval_seconds=15.0,
                    miner_submission_health_timeout_seconds=10.0,
                    hf_cache_dir="",
                    hf_activitynet_filename="",
                    evaluation_round_blocks=0,
                ),
            )

        self.assertTrue(mock_subtensor.execute.called)
        call_args = mock_subtensor.execute.call_args
        self.assertIsNotNone(call_args)
        intent = call_args.args[0]
        self.assertEqual(intent.netuid, 1)
        self.assertEqual(intent.uids, [0, 1, 2])
        self.assertEqual(len(intent.weights), 3)
        self.assertIs(call_args.args[1], mock_wallet)
        self.assertTrue(call_args.kwargs["wait_for_inclusion"])
        self.assertFalse(call_args.kwargs["wait_for_finalization"])

    @patch("chronoseek.validator.task_gen.ActivityNetTaskGenerator")
    @patch("chronoseek.validator.forward.run_step")
    async def test_validator_loop_paces_rounds_by_block_not_wall_clock(
        self, mock_run_step, mock_task_gen
    ):
        """A round starts every evaluation_round_blocks since the previous
        round's start, or immediately if a slow round already blew past that
        boundary - not on a wall-clock timer."""
        mock_subtensor = MagicMock()
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        stop_event = MagicMock()

        mock_metagraph.n = 1
        mock_metagraph.hotkeys = ["h1"]
        mock_metagraph.I = [1.0]

        mock_subtensor.subnets.info.return_value.tempo = 1000
        mock_subtensor.execute.return_value = True
        # Round due at block 100, not due at 105, due again at 111 (one block
        # past the 110 boundary - starts immediately rather than waiting for
        # a later "clean" boundary), not due at 115, due again at 121.
        mock_subtensor.block.side_effect = [100, 105, 111, 115, 121]
        stop_event.is_set.side_effect = [False] * 5 + [True]

        mock_run_step.return_value = []

        with patch("asyncio.sleep", new_callable=AsyncMock), patch(
            "validator.MinerSubmissionResolver"
        ) as mock_resolver_cls, patch(
            "validator.refresh_responsive_miners_from_submissions", new_callable=AsyncMock
        ) as mock_refresh_responsive:
            mock_resolver_cls.return_value = MagicMock()

            async def refresh_side_effect(*args, **kwargs):
                refresh_runtime = kwargs["runtime"]
                with refresh_runtime.responsive_lock:
                    refresh_runtime.responsive_uids = {0}
                    refresh_runtime.miner_endpoints = {
                        0: "https://runtime-0.example.com"
                    }
                    refresh_runtime.responsive_initialized = True
                    refresh_runtime.responsive_last_refresh_at = 1.0
                return {0}

            mock_refresh_responsive.side_effect = refresh_side_effect
            runtime = ValidatorRuntimeState(
                wallet=mock_wallet,
                metagraph=mock_metagraph,
                scores=seed_scores_from_metagraph(mock_metagraph),
                score_lock=threading.Lock(),
            )
            await run_validator_loop(
                mock_subtensor,
                runtime,
                netuid=1,
                stop_event=stop_event,
                last_heartbeat=[0],
                config=MagicMock(
                    task_dataset_path="",
                    task_split="validation",
                    require_accessible_videos=False,
                    task_max_sampling_attempts=10,
                    miner_request_timeout_seconds=60.0,
                    miner_emission_burn_percent=0.0,
                    video_availability_cache_path="",
                    accessible_video_cache_path="",
                    inaccessible_video_cache_path="",
                    video_availability_cache_ttl_hours=24,
                    video_availability_timeout=5,
                    miner_submission_refresh_interval_seconds=15.0,
                    miner_submission_health_timeout_seconds=10.0,
                    hf_cache_dir="",
                    hf_activitynet_filename="",
                    evaluation_round_blocks=10,
                ),
            )

        self.assertEqual(mock_run_step.call_count, 3)

    @patch("chronoseek.validator.task_gen.ActivityNetTaskGenerator")
    @patch("chronoseek.validator.forward.run_step")
    async def test_validator_loop_skips_unhealthy_submission_runtimes(
        self,
        mock_run_step,
        mock_task_gen,
    ):
        mock_subtensor = MagicMock()
        mock_subtensor.subnets.info.return_value.tempo = 100
        mock_subtensor.block.return_value = 1
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.n = 1
        mock_metagraph.hotkeys = ["h1"]
        mock_metagraph.I = [1.0]
        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]

        with patch("asyncio.sleep", new_callable=AsyncMock), patch(
            "validator.MinerSubmissionResolver"
        ) as mock_resolver_cls, patch(
            "validator.refresh_responsive_miners_from_submissions", new_callable=AsyncMock
        ) as mock_refresh_responsive:
            mock_resolver = MagicMock()
            mock_resolver.get_submissions = AsyncMock(
                return_value={
                    "h1": MinerSubmission(
                        hotkey="h1", endpoint="https://runtime-0.example.com"
                    )
                }
            )
            mock_resolver_cls.return_value = mock_resolver

            async def refresh_side_effect(*args, **kwargs):
                refresh_runtime = kwargs["runtime"]
                with refresh_runtime.responsive_lock:
                    refresh_runtime.responsive_uids = set()
                    refresh_runtime.miner_endpoints = {}
                    refresh_runtime.responsive_initialized = True
                    refresh_runtime.responsive_last_refresh_at = 1.0
                return set()

            mock_refresh_responsive.side_effect = refresh_side_effect
            runtime = ValidatorRuntimeState(
                wallet=mock_wallet,
                metagraph=mock_metagraph,
                scores=seed_scores_from_metagraph(mock_metagraph),
                score_lock=threading.Lock(),
            )

            await run_validator_loop(
                mock_subtensor,
                runtime,
                netuid=1,
                stop_event=stop_event,
                last_heartbeat=[0],
                config=MagicMock(
                    task_dataset_path="",
                    task_split="validation",
                    require_accessible_videos=False,
                    task_max_sampling_attempts=10,
                    miner_request_timeout_seconds=60.0,
                    miner_emission_burn_percent=0.0,
                    video_availability_cache_path="",
                    accessible_video_cache_path="",
                    inaccessible_video_cache_path="",
                    video_availability_cache_ttl_hours=24,
                    video_availability_timeout=5,
                    miner_submission_refresh_interval_seconds=15.0,
                    miner_submission_health_timeout_seconds=10.0,
                    hf_cache_dir="",
                    hf_activitynet_filename="",
                    evaluation_round_blocks=0,
                ),
            )

        mock_run_step.assert_not_called()

    @patch("chronoseek.validator.task_gen.ActivityNetTaskGenerator")
    async def test_validator_loop_syncs_metagraph_growth_from_incentives(
        self,
        mock_task_gen,
    ):
        mock_subtensor = MagicMock()
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        stop_event = MagicMock()

        mock_metagraph.n = 2
        mock_metagraph.hotkeys = ["h1", "h2"]
        mock_metagraph.I = [0.4, 0.6]

        synced_metagraph = MagicMock()
        synced_metagraph.n = 3
        synced_metagraph.hotkeys = ["h1", "h2", "h3"]
        synced_metagraph.I = [0.4, 0.6, 0.9]
        synced_metagraph.uids = [0, 1, 2]

        mock_subtensor.subnets.info.return_value.tempo = 100
        mock_subtensor.block.side_effect = [0, 1]
        stop_event.is_set.side_effect = [False, True]

        with patch("asyncio.sleep", new_callable=AsyncMock), patch(
            "validator.sync_runtime_metagraph"
        ) as mock_sync_runtime_metagraph, patch(
            "validator.refresh_responsive_miners_from_submissions", new_callable=AsyncMock
        ) as mock_refresh_responsive_miners:
            def sync_runtime_metagraph_side_effect(subtensor, runtime, netuid):
                runtime.metagraph = synced_metagraph
                runtime.scores = np.array([0.4, 0.6, 0.9])
                return synced_metagraph

            mock_sync_runtime_metagraph.side_effect = sync_runtime_metagraph_side_effect
            mock_refresh_responsive_miners.return_value = {0, 1, 2}
            runtime = ValidatorRuntimeState(
                wallet=mock_wallet,
                metagraph=mock_metagraph,
                scores=seed_scores_from_metagraph(mock_metagraph),
                score_lock=threading.Lock(),
            )
            await run_validator_loop(
                mock_subtensor,
                runtime,
                netuid=1,
                stop_event=stop_event,
                last_heartbeat=[0],
                config=MagicMock(
                    task_dataset_path="",
                    task_split="validation",
                    require_accessible_videos=False,
                    task_max_sampling_attempts=10,
                    miner_request_timeout_seconds=60.0,
                    miner_emission_burn_percent=0.0,
                    video_availability_cache_path="",
                    accessible_video_cache_path="",
                    inaccessible_video_cache_path="",
                    video_availability_cache_ttl_hours=24,
                    video_availability_timeout=5,
                    miner_submission_refresh_interval_seconds=15.0,
                    miner_submission_health_timeout_seconds=10.0,
                    hf_cache_dir="",
                    hf_activitynet_filename="",
                    evaluation_round_blocks=0,
                ),
            )

        np.testing.assert_allclose(runtime.scores, np.array([0.4, 0.6, 0.9]))

    @patch("chronoseek.chutes.runtime.httpx.AsyncClient")
    async def test_submission_refresh_sets_responsive_uids_from_metadata_and_health(
        self,
        mock_async_client_cls,
    ):
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["hk-0", "hk-1"]
        mock_metagraph.uids = [0, 1]
        mock_metagraph.n = 2

        submission = MagicMock()
        submission.endpoint = "https://runtime.example.com"
        submission.chute_slug = None

        resolver = MagicMock()
        resolver.get_submissions = AsyncMock(return_value={"hk-1": submission})
        resolver.get_duplicate_hotkeys.return_value = set()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}
        client = AsyncMock()
        client.get.return_value = response
        mock_async_client_cls.return_value.__aenter__.return_value = client

        runtime = ValidatorRuntimeState(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            scores=np.array([0.0, 1.0]),
            score_lock=threading.Lock(),
        )

        responsive_uids = await refresh_responsive_miners_from_submissions(
            runtime=runtime,
            subtensor=MagicMock(),
            netuid=1,
            submission_resolver=resolver,
            chutes_base_domain="chutes.ai",
            health_timeout_seconds=10,
            provider_headers={"Authorization": "Bearer secret"},
        )

        self.assertEqual(responsive_uids, {1})
        self.assertEqual(runtime.responsive_uids, {1})
        self.assertEqual(
            runtime.miner_endpoints,
            {1: "https://runtime.example.com"},
        )
        self.assertEqual(
            client.get.call_args.args[0],
            "https://runtime.example.com/health",
        )
        self.assertEqual(
            client.get.call_args.kwargs["headers"],
            {"Authorization": "Bearer secret"},
        )

    @patch("chronoseek.chutes.runtime.httpx.AsyncClient")
    async def test_submission_refresh_excludes_unhealthy_runtime(
        self,
        mock_async_client_cls,
    ):
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["hk-0", "hk-1"]
        mock_metagraph.uids = [0, 1]
        mock_metagraph.n = 2

        submission = MagicMock()
        submission.endpoint = "https://runtime.example.com"
        submission.chute_slug = None

        resolver = MagicMock()
        resolver.get_submissions = AsyncMock(return_value={"hk-1": submission})
        resolver.get_duplicate_hotkeys.return_value = set()
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": False}
        client = AsyncMock()
        client.get.return_value = response
        mock_async_client_cls.return_value.__aenter__.return_value = client

        runtime = ValidatorRuntimeState(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            scores=np.array([0.0, 1.0]),
            score_lock=threading.Lock(),
        )

        responsive_uids = await refresh_responsive_miners_from_submissions(
            runtime=runtime,
            subtensor=MagicMock(),
            netuid=1,
            submission_resolver=resolver,
            chutes_base_domain="chutes.ai",
            health_timeout_seconds=10,
        )

        self.assertEqual(responsive_uids, set())
        self.assertEqual(runtime.responsive_uids, set())
        self.assertEqual(runtime.miner_endpoints, {})

    @patch("chronoseek.chutes.runtime.httpx.AsyncClient")
    async def test_submission_refresh_records_duplicate_submitter_disqualification(
        self,
        mock_async_client_cls,
    ):
        mock_wallet = MagicMock()
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["hk-0", "hk-1"]
        mock_metagraph.uids = [0, 1]
        mock_metagraph.n = 2

        resolver = MagicMock()
        resolver.get_submissions = AsyncMock(return_value={})
        resolver.get_duplicate_hotkeys.return_value = {"hk-1"}
        mock_async_client_cls.return_value.__aenter__.return_value = AsyncMock()

        runtime = ValidatorRuntimeState(
            wallet=mock_wallet,
            metagraph=mock_metagraph,
            scores=np.array([0.4, 0.6]),
            score_lock=threading.Lock(),
        )

        responsive_uids = await refresh_responsive_miners_from_submissions(
            runtime=runtime,
            subtensor=MagicMock(),
            netuid=1,
            submission_resolver=resolver,
            chutes_base_domain="chutes.ai",
            health_timeout_seconds=10,
        )

        self.assertEqual(responsive_uids, set())
        self.assertEqual(runtime.disqualified_uids, {1})
        self.assertEqual(runtime.responsive_uids, set())
        self.assertEqual(runtime.miner_endpoints, {})


if __name__ == "__main__":
    unittest.main()


class TestValidatorForward(unittest.IsolatedAsyncioTestCase):

    @patch("chronoseek.validator.forward.generate_header")
    async def test_query_miner_serializes_request_payload_as_json(
        self, mock_generate_header
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey = MagicMock()

        request = VideoSearchRequest(
            request_id="req-1",
            video={"url": "https://example.com/video.mp4"},
            query="a person is speaking",
            top_k=3,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "protocol_version": "2026-04-10",
            "request_id": "req-1",
            "status": "completed",
            "results": [],
        }

        client = AsyncMock()
        client.post.return_value = mock_response
        mock_generate_header.return_value = {"X-Test": "1"}

        await query_miner(
            client,
            1,
            "hk-1",
            "https://runtime.example.com",
            request,
            mock_wallet,
        )

        header_body = mock_generate_header.call_args.args[1]
        self.assertEqual(header_body["video"]["url"], "https://example.com/video.mp4")
        self.assertIsInstance(header_body["video"]["url"], str)

        sent_json = client.post.call_args.kwargs["json"]
        self.assertEqual(sent_json["video"]["url"], "https://example.com/video.mp4")
        self.assertIsInstance(sent_json["video"]["url"], str)

    async def test_run_step_retries_when_no_accessible_video_is_found(self):
        from chronoseek.validator.forward import run_step

        task_gen = MagicMock()
        task_gen.generate_task.side_effect = [
            RuntimeError("Unable to generate a validator task from an accessible video."),
            (
                "https://example.com/video.mp4",
                "a person waves",
                [(1.0, 2.0)],
            ),
        ]
        task_gen.refresh_video_lookup.return_value = 3

        metagraph = MagicMock()
        metagraph.uids = []
        wallet = MagicMock()
        client = AsyncMock()

        scores = await run_step(task_gen, metagraph, wallet, client)

        self.assertEqual(scores, [])
        task_gen.refresh_video_lookup.assert_called_once()
        self.assertEqual(task_gen.generate_task.call_count, 2)
