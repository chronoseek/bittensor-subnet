#!/usr/bin/env python3
"""
Send a signed test request to a running ChronoSeek miner.

Usage:
    poetry run python scripts/test_miner_search.py
"""

import argparse
import json
import time
import uuid

import bittensor as bt
import requests

from chronoseek.epistula import generate_header

DEFAULT_SMOKE_VIDEO_URL = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
DEFAULT_SMOKE_QUERY = "a person is on screen"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ChronoSeek miner /search endpoint"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/search",
        help="Miner search endpoint",
    )
    parser.add_argument(
        "--video-url",
        default=None,
        help="Public video URL for testing",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Natural language search query",
    )
    parser.add_argument(
        "--scenario",
        choices=["smoke", "speech"],
        default="smoke",
        help="Use a built-in test scenario when --video-url/--query are not provided.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k results to request (1-20)",
    )
    parser.add_argument(
        "--wallet-name",
        default="testnet-miner",
        help="Wallet name used to sign Epistula headers",
    )
    parser.add_argument(
        "--wallet-hotkey",
        default="h1",
        help="Hotkey used to sign Epistula headers",
    )
    parser.add_argument(
        "--wallet-path",
        default="~/.bittensor/wallets/",
        help="Wallet path used to load keypair",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_k < 1 or args.top_k > 20:
        print("Invalid --top-k value. Must be between 1 and 20.")
        return 2

    if args.video_url and args.query:
        video_url = args.video_url
        query = args.query
    elif args.scenario == "speech":
        print(
            "The 'speech' scenario requires an explicit --video-url and --query "
            "for a video that actually contains speech."
        )
        return 2
    else:
        video_url = DEFAULT_SMOKE_VIDEO_URL
        query = DEFAULT_SMOKE_QUERY

    request_id = f"local-test-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    body = {
        "protocol_version": "2026-04-10",
        "request_id": request_id,
        "video": {"url": video_url},
        "query": query,
        "top_k": args.top_k,
    }

    wallet = bt.Wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path,
    )
    headers = generate_header(wallet.hotkey, body)

    print(f"Sending signed request to: {args.endpoint}")
    print(f"Scenario:   {args.scenario}")
    print(f"request_id: {request_id}")
    print(f"video_url:  {video_url}")
    print(f"query:      {query}")
    response = requests.post(
        args.endpoint,
        json=body,
        headers=headers,
        timeout=args.timeout,
    )

    print(f"HTTP {response.status_code}")
    try:
        parsed = response.json()
        print(json.dumps(parsed, indent=2))
    except ValueError:
        print(response.text)

    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
