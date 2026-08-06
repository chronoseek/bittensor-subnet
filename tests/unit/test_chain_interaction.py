import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from miner import get_config, main as miner_main
from chronoseek.chain.submissions import PERMANENT_SUBMISSION_ERROR


class TestChainInteraction(unittest.TestCase):
    def test_miner_config_defaults_to_finney_netuid_20(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "argv", ["miner.py"]):
                config = get_config()

        self.assertEqual(config.subtensor.network, "finney")
        self.assertEqual(config.netuid, 20)

    def test_miner_config_accepts_environment_chain_override(self):
        with patch.dict(
            os.environ,
            {"NETWORK": "test", "NETUID": "298"},
            clear=True,
        ):
            with patch.object(sys, "argv", ["miner.py"]):
                config = get_config()

        self.assertEqual(config.subtensor.network, "test")
        self.assertEqual(config.netuid, 298)

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

    def test_miner_config_accepts_network_alias(self):
        test_args = [
            "miner.py",
            "--network",
            "local",
        ]

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "argv", test_args):
                config = get_config()

        self.assertEqual(config.subtensor.network, "local")

    def test_miner_config_accepts_check_existing(self):
        test_args = [
            "miner.py",
            "--check-existing",
        ]

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys, "argv", test_args):
                config = get_config()

        self.assertTrue(config.check_existing)

    def test_miner_submission_enforcement_defaults_to_enabled(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys, "argv", ["miner.py"]
        ):
            config = get_config()

        self.assertTrue(config.enforce_one_hotkey_one_submission)

    def test_miner_submission_enforcement_can_be_disabled_by_environment(self):
        with patch.dict(
            os.environ,
            {"ENFORCE_ONE_HOTKEY_ONE_SUBMISSION": "0"},
            clear=True,
        ), patch.object(sys, "argv", ["miner.py"]):
            config = get_config()

        self.assertFalse(config.enforce_one_hotkey_one_submission)

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_miner_commits_runtime_submission(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor.endpoint = "wss://test.example"
        mock_subtensor.subnets.commitments.return_value = {}
        mock_subtensor.submit_call.return_value = True
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph.neurons = []
        mock_subtensor.subnets.metagraph.return_value = mock_metagraph

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

        with patch.object(sys, "argv", test_args), patch("builtins.print"), patch(
            "chronoseek.chain.submissions.get_encrypted_commitment",
            return_value=(b"encrypted", 12345),
        ):
            exit_code = miner_main()

        self.assertEqual(exit_code, 0)
        mock_wallet_cls.assert_called()
        mock_subtensor_cls.assert_called()
        mock_subtensor.subnets.metagraph.assert_called_once_with(
            298, commitments=False
        )
        mock_subtensor.submit_call.assert_called_once()
        call_args = mock_subtensor.submit_call.call_args
        call = call_args.args[0]
        self.assertEqual(call.module, "Commitments")
        self.assertEqual(call.function, "set_commitment")
        self.assertEqual(call.params["netuid"], 298)
        self.assertEqual(
            call.params["info"]["fields"][0][0]["TimelockEncrypted"],
            {"encrypted": b"encrypted", "reveal_round": 12345},
        )
        self.assertEqual(call_args.kwargs["signer"], "hotkey")

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_miner_check_existing_prints_commitments_without_submitting(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor.endpoint = "wss://test.example"
        mock_subtensor.subnets.commitments.return_value = {
            "5FakeAddress": SimpleNamespace(
                revealed=[(
                    123,
                    '{"chute_slug":"chronoseek-runtime","runtime":"chutes"}',
                )],
                encrypted=True,
            )
        }
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph.neurons = []
        mock_subtensor.subnets.metagraph.return_value = mock_metagraph

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--check-existing",
        ]

        with patch.object(sys, "argv", test_args), patch(
            "builtins.print"
        ) as mock_print:
            exit_code = miner_main()

        self.assertEqual(exit_code, 0)
        mock_subtensor.subnets.commitments.assert_called_once_with(298)
        mock_subtensor.submit_call.assert_not_called()
        printed = json.loads(mock_print.call_args.args[0])
        self.assertEqual(printed["hotkey"], "5FakeAddress")
        self.assertEqual(printed["netuid"], 298)
        self.assertTrue(printed["registered"])
        self.assertEqual(printed["commitments"][0]["block"], 123)
        self.assertEqual(
            printed["commitments"][0]["data"]["chute_slug"],
            "chronoseek-runtime",
        )

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_miner_rejects_missing_subnet_on_selected_network(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "finney"
        mock_subtensor.endpoint = "wss://entrypoint-finney.opentensor.ai:443"
        mock_subtensor.subnets.metagraph.return_value = None
        mock_subtensor_cls.return_value = mock_subtensor

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--network",
            "finney",
            "--chute-slug",
            "chronoseek-runtime",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"), patch(
            "miner.logger.error"
        ) as mock_log_error:
            exit_code = miner_main()

        self.assertEqual(exit_code, 1)
        mock_subtensor.subnets.metagraph.assert_called_once_with(
            298, commitments=False
        )
        self.assertIn(
            "Subnet netuid=298 does not exist on network=finney",
            mock_log_error.call_args.args[0],
        )

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_miner_rejects_chute_id_only_submission_until_resolver_exists(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor.endpoint = "wss://test.example"
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph.neurons = []
        mock_subtensor.subnets.metagraph.return_value = mock_metagraph

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
        mock_subtensor.submit_call.assert_not_called()

    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    def test_miner_rejects_second_submission_for_same_hotkey(
        self,
        mock_subtensor_cls,
        mock_wallet_cls,
    ):
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet

        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor.endpoint = "wss://test.example"
        mock_subtensor.subnets.commitments.return_value = {
            "5FakeAddress": SimpleNamespace(
                revealed=[(100, "{}")],
                encrypted=True,
            )
        }
        mock_subtensor_cls.return_value = mock_subtensor

        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph.neurons = []
        mock_subtensor.subnets.metagraph.return_value = mock_metagraph

        test_args = [
            "miner.py",
            "--netuid",
            "298",
            "--chute-slug",
            "replacement-runtime",
        ]

        with patch.object(sys, "argv", test_args), patch("builtins.print"), patch(
            "miner.logger.error"
        ) as mock_log_error:
            exit_code = miner_main()

        self.assertEqual(exit_code, 1)
        mock_subtensor.submit_call.assert_not_called()
        mock_log_error.assert_any_call(PERMANENT_SUBMISSION_ERROR)

if __name__ == "__main__":
    unittest.main()
