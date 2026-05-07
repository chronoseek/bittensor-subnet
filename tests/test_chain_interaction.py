import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from miner import main as miner_main
from scripts.commit_v2_submission import main_async as commit_v2_main_async


class TestChainInteraction(unittest.TestCase):
    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    @patch("bittensor.Metagraph")
    def test_miner_commits_runtime_submission(
        self,
        mock_metagraph_cls,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor.set_reveal_commitment.return_value = True
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph_cls.return_value = mock_metagraph

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--wallet.name",
            "test_wallet",
            "--wallet.hotkey",
            "test_hotkey",
            "--chute-slug",
            "chronoseek-runtime",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"):
            exit_code = miner_main()

        self.assertEqual(exit_code, 0)
        mock_wallet_cls.assert_called()
        mock_subtensor_cls.assert_called()
        mock_metagraph_cls.assert_called_with(netuid=298, network="test", sync=False)
        mock_metagraph.sync.assert_called_with(subtensor=mock_subtensor)
        mock_subtensor.set_reveal_commitment.assert_called_once()

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    @patch("bittensor.Metagraph")
    def test_miner_rejects_chute_id_only_submission_until_resolver_exists(
        self,
        mock_metagraph_cls,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph_cls.return_value = mock_metagraph

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--chute-id",
            "chute-deployment-id",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"):
            exit_code = miner_main()

        self.assertEqual(exit_code, 1)
        mock_subtensor.set_reveal_commitment.assert_not_called()

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_commit_script_rejects_chute_id_only_submission_until_resolver_exists(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        test_args = [
            "commit_v2_submission.py",
            "--netuid",
            "298",
            "--chute-id",
            "chute-deployment-id",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"):
            exit_code = asyncio.run(commit_v2_main_async())

        self.assertEqual(exit_code, 1)
        mock_subtensor_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
