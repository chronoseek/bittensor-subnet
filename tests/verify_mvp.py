import argparse
import os
import sys
import uuid
from pathlib import Path

import bittensor as bt
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from chronoseek.miner.logic import MinerLogic, SearchPipelineError
from chronoseek.protocol_models import VideoSearchRequest, VideoSearchResponse
from chronoseek.scoring import calculate_iou, score_response
from chronoseek.validator.task_gen import ActivityNetTaskGenerator

DEFAULT_BOOTSTRAP_DATASET_PATH = str(
    Path(__file__).resolve().parent.parent
    / "chronoseek"
    / "validator"
    / "data"
    / "activitynet_bootstrap.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="ChronoSeek MVP end-to-end verifier")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("CHRONOSEEK_VERIFY_DATASET_PATH", DEFAULT_BOOTSTRAP_DATASET_PATH),
        help="Local ActivityNet manifest/JSON path for verification. Defaults to the bootstrap dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=os.getenv("CHRONOSEEK_VERIFY_SPLIT", "validation"),
        help="Dataset split to use when loading from Hugging Face.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.getenv("CHRONOSEEK_VERIFY_ATTEMPTS", "3")),
        help="Number of random task attempts before failing verification.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Fail unless the final score passes the strict IoU threshold.",
    )
    return parser.parse_args()


def build_task_generator(args):
    return ActivityNetTaskGenerator(
        dataset_path=args.dataset_path,
        split=args.split,
    )


def print_header(title: str):
    print(f"\n=== {title} ===")


def print_task(task_number: int, video_url: str, query: str, ground_truth):
    print(f"Attempt: {task_number}")
    print(f"Video:   {video_url}")
    print(f"Query:   {query}")
    print(f"GT:      {ground_truth[0]:.2f}s -> {ground_truth[1]:.2f}s")


def verify_protocol_request(video_url: str, query: str) -> VideoSearchRequest:
    request = VideoSearchRequest(
        request_id=f"verify-{uuid.uuid4()}",
        video={"url": video_url},
        query=query,
        top_k=5,
    )
    print(f"Request ID: {request.request_id}")
    print("Protocol request validation: ok")
    return request


def verify_protocol_response(request_id: str, results) -> VideoSearchResponse:
    response = VideoSearchResponse(
        request_id=request_id,
        status="completed",
        results=results,
        miner_metadata={"source": "verify_mvp"},
    )
    print("Protocol response validation: ok")
    return response


def run_single_attempt(task_gen: ActivityNetTaskGenerator, miner: MinerLogic, task_number: int):
    video_url, query, ground_truth = task_gen.generate_task()

    print_header("Task Generation")
    print_task(task_number, video_url, query, ground_truth)

    request = verify_protocol_request(video_url, query)

    print_header("Miner Search")
    results = miner.search(request.video_url, request.query)
    response = verify_protocol_response(request.request_id, results)

    if not response.results:
        raise RuntimeError("Miner returned an empty result list.")

    best = response.results[0]
    raw_iou = calculate_iou(best.start, best.end, ground_truth[0], ground_truth[1])
    binary_score = score_response(response.results, ground_truth, latency=0.0)

    print(f"Top result: {best.start:.2f}s -> {best.end:.2f}s")
    print(f"Confidence: {best.confidence:.4f}")
    print(f"Raw IoU:    {raw_iou:.4f}")
    print(f"Score:      {binary_score:.1f}")

    return {
        "request": request,
        "response": response,
        "ground_truth": ground_truth,
        "raw_iou": raw_iou,
        "score": binary_score,
    }


def main():
    args = parse_args()

    bt.logging.set_info(True)
    print("ChronoSeek MVP Verification")
    print(f"Split: {args.split}")
    print(f"Dataset path: {args.dataset_path}")

    try:
        task_gen = build_task_generator(args)
    except Exception as exc:
        print_header("Setup Failure")
        print(str(exc))
        return 1

    try:
        miner = MinerLogic()
    except Exception as exc:
        print_header("Miner Initialization Failure")
        print(str(exc))
        return 1

    last_error = None
    for attempt_number in range(1, args.attempts + 1):
        try:
            result = run_single_attempt(task_gen, miner, attempt_number)
            passed = result["score"] == 1.0

            print_header("Verification Summary")
            print("End-to-end flow: ok")
            print(f"Strict IoU pass: {'yes' if passed else 'no'}")

            if args.require_pass and not passed:
                print("Verification failed: strict IoU pass was required.")
                return 1

            return 0
        except SearchPipelineError as exc:
            last_error = f"{exc.code}: {exc.message}"
            print_header("Attempt Failed")
            print(last_error)
        except Exception as exc:
            last_error = str(exc)
            print_header("Attempt Failed")
            print(last_error)

    print_header("Verification Failed")
    if last_error:
        print(last_error)
    print(f"All {args.attempts} attempts failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
