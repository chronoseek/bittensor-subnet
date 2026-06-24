import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from miner import get_config, main as miner_main
from chronoseek.chain.submissions import PERMANENT_SUBMISSION_ERROR


class TestChainInteraction(unittest.TestCase):
    def test_miner_config_parses_wallet_cli_args_by_default(self):
        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--wallet.name",
            "testnet-miner",
            "--wallet.hotkey",
            "h1",
            "--subtensor.network",
            "test",
        ]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BT_NO_PARSE_CLI_ARGS", None)
            with patch.object(sys, "argv", test_args):
                config = get_config()

        self.assertEqual(config.netuid, 298)
        self.assertEqual(config.wallet.name, "testnet-miner")
        self.assertEqual(config.wallet.hotkey, "h1")
        self.assertEqual(config.subtensor.network, "test")

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
    @patch("bittensor.Metagraph")
    def test_miner_rejects_second_submission_for_same_hotkey(
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
        mock_subtensor.get_all_revealed_commitments.return_value = {
            "5FakeAddress": ((100, "{}"),)
        }
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph_cls.return_value = mock_metagraph

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--chute-slug",
            "replacement-runtime",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"), patch(
            "bittensor.logging.error"
        ) as mock_log_error:
            exit_code = miner_main()

        self.assertEqual(exit_code, 1)
        mock_subtensor.set_reveal_commitment.assert_not_called()
        mock_log_error.assert_any_call(PERMANENT_SUBMISSION_ERROR)

if __name__ == "__main__":
    unittest.main()
