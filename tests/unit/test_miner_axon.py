import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import miner_axon


class TestPublishAxon(unittest.TestCase):
    def test_publish_axon_requires_external_ip(self):
        config = SimpleNamespace(
            axon_external_ip=None,
            axon_external_port=None,
            axon_port=8091,
            netuid=298,
        )

        with self.assertRaises(ValueError):
            miner_axon.publish_axon(config, MagicMock(), MagicMock())

    def test_publish_axon_executes_serve_axon_intent(self):
        config = SimpleNamespace(
            axon_external_ip="1.2.3.4",
            axon_external_port=None,
            axon_port=8091,
            netuid=298,
        )
        wallet = MagicMock()
        subtensor = MagicMock()
        subtensor.execute.return_value = SimpleNamespace(success=True, message="ok")

        miner_axon.publish_axon(config, wallet, subtensor)

        subtensor.execute.assert_called_once()
        intent, called_wallet = subtensor.execute.call_args.args
        self.assertEqual(called_wallet, wallet)
        self.assertEqual(intent.netuid, 298)
        self.assertEqual(intent.ip, "1.2.3.4")
        self.assertEqual(intent.port, 8091)

    def test_publish_axon_uses_external_port_when_given(self):
        config = SimpleNamespace(
            axon_external_ip="1.2.3.4",
            axon_external_port=9500,
            axon_port=8091,
            netuid=298,
        )
        subtensor = MagicMock()
        subtensor.execute.return_value = SimpleNamespace(success=True, message="ok")

        miner_axon.publish_axon(config, MagicMock(), subtensor)

        intent = subtensor.execute.call_args.args[0]
        self.assertEqual(intent.port, 9500)

    def test_publish_axon_raises_on_chain_failure(self):
        config = SimpleNamespace(
            axon_external_ip="1.2.3.4",
            axon_external_port=None,
            axon_port=8091,
            netuid=298,
        )
        subtensor = MagicMock()
        subtensor.execute.return_value = SimpleNamespace(
            success=False, message="serving rate limit exceeded"
        )

        with self.assertRaises(RuntimeError):
            miner_axon.publish_axon(config, MagicMock(), subtensor)


if __name__ == "__main__":
    unittest.main()
