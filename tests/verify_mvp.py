import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chronoseek.validator.task_gen import ActivityNetTaskGenerator
from chronoseek.miner.logic import MinerLogic
from chronoseek.scoring import score_response
import bittensor as bt

# Configure logging
logging.basicConfig(level=logging.INFO)
bt.logging.set_debug(True)


def main():
    print("--- Starting MVP Verification ---")

    # 1. Test Task Generation
    print("\n1. Testing Task Generator...")
    task_gen = ActivityNetTaskGenerator()
    video_url, query, ground_truth = task_gen.generate_task()

    # Override for reliability in testing environment (Big Buck Bunny is usually fastest)
    video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    query = "the big rabbit wakes up from his hole"
    ground_truth = (10.0, 25.0)

    print(f"   Video: {video_url}")
    print(f"   Query: {query}")
    print(f"   GT: {ground_truth}")

    # 2. Test Miner Logic (CLIP)
    print("\n2. Testing Miner Logic (This may take time to download/process)...")
    miner = MinerLogic()

    results = miner.search(video_url, query)
    print(f"   Miner Results: {results}")

    if not results:
        print("   [!] Miner returned no results.")
        return

    # 3. Test Scoring
    print("\n3. Testing Scoring (Strict IoU > 0.5)...")
    score = score_response(results, ground_truth, latency=5.0)
    print(f"   Score: {score}")

    if score == 1.0:
        print("   [SUCCESS] Perfect Match!")
    elif score == 0.0:
        print("   [FAIL] No overlap or IoU < 0.5")

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    main()
