import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import bittensor as bt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock uvicorn before importing miner to avoid starting server
with patch("uvicorn.run"):
    from miner import main as miner_main

class TestChainInteraction(unittest.TestCase):
    
    @patch("bittensor.Wallet")
    @patch("bittensor.Subtensor")
    @patch("bittensor.Metagraph")
    @patch("bittensor.Axon")
    @patch("uvicorn.run")
    @patch("chronoseek.miner.logic.MinerLogic")
    def test_miner_serve_axon(self, mock_logic, mock_uvicorn, mock_axon_cls, mock_metagraph_cls, mock_subtensor_cls, mock_wallet_cls):
        """
        Test that miner.py correctly initializes wallet, subtensor, and calls serve_axon.
        """
        # Setup Mocks
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "5FakeAddress"
        mock_wallet_cls.return_value = mock_wallet
        
        mock_subtensor = MagicMock()
        mock_subtensor.network = "test"
        mock_subtensor_cls.return_value = mock_subtensor
        
        mock_metagraph = MagicMock()
        # Simulate miner being registered
        mock_metagraph.hotkeys = ["5FakeAddress"]
        mock_metagraph_cls.return_value = mock_metagraph
        
        mock_axon_instance = MagicMock()
        mock_axon_cls.return_value = mock_axon_instance
        
        # Mock CLI args to avoid parsing real sys.argv
        test_args = [
            "miner.py", 
            "--netuid", "298", 
            "--wallet.name", "test_wallet",
            "--wallet.hotkey", "test_hotkey",
            "--axon.port", "8091"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Run miner main
            miner_main()
            
            # Verifications
            
            # 1. Check Wallet Init
            mock_wallet_cls.assert_called()
            
            # 2. Check Subtensor Init
            mock_subtensor_cls.assert_called()
            
            # 3. Check Metagraph Init
            mock_metagraph_cls.assert_called_with(netuid=298, network="test")
            
            # 4. Check Axon Creation (using passed port 8091)
            # Note: We prioritize axon.port, so it should be used
            mock_axon_cls.assert_called_with(wallet=mock_wallet, port=8091)
            
            # 5. Check serve_axon call
            # This is the critical chain interaction we added
            mock_subtensor.serve_axon.assert_called_with(
                netuid=298,
                axon=mock_axon_instance
            )
            
            # 6. Check Uvicorn start
            mock_uvicorn.assert_called_with(
                unittest.mock.ANY, # The FastAPI app
                host="0.0.0.0",
                port=8091
            )
            
            print("✅ Miner Chain Interaction Test Passed: serve_axon called correctly.")

if __name__ == "__main__":
    unittest.main()
