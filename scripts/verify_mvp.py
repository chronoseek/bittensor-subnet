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
from chronoseek.scoring import (
    STRICT_IOU_THRESHOLD,
    best_iou,
    calculate_iou,
    passes_strict_iou,
    score_response,
)
from chronoseek.validator.task_gen import ActivityNetTaskGenerator

DEFAULT_SMOKE_TEST_DATASET_PATH = str(
    Path(__file__).resolve().parent.parent
    / "chronoseek"
    / "validator"
    / "data"
    / "smoke_test_tasks.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="ChronoSeek MVP end-to-end verifier")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_SMOKE_TEST_DATASET_PATH,
        help="Local smoke-test manifest/JSON path for verification. Defaults to the curated smoke-test dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use when loading from Hugging Face.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of random task attempts before failing verification.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=f"Fail unless the final score passes the strict IoU threshold ({STRICT_IOU_THRESHOLD:.2f}).",
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default="",
        help="Optional custom video URL for direct end-to-end miner verification.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional custom query used with --video-url.",
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Skip IoU quality checks. Useful when running a custom video/query without ground truth.",
    )
    return parser.parse_args()


def build_task_generator(args):
    return ActivityNetTaskGenerator(
        dataset_path=args.dataset_path,
        split=args.split,
    )


def print_header(title: str):
    print(f"\n=== {title} ===")


def print_task(task_number: int, video_url: str, query: str, ground_truths):
    print(f"Attempt: {task_number}")
    print(f"Video:   {video_url}")
    print(f"Query:   {query}")
    print("GT:")
    for start, end in ground_truths:
        print(f"  - {start:.2f}s -> {end:.2f}s")


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


def run_single_attempt(
    task_gen: ActivityNetTaskGenerator | None,
    miner: MinerLogic,
    task_number: int,
    custom_video_url: str = "",
    custom_query: str = "",
    skip_quality_check: bool = False,
):
    if custom_video_url:
        video_url = custom_video_url
        query = custom_query
        ground_truths = []
    else:
        if task_gen is None:
            raise RuntimeError("Task generator is required when no custom video URL is provided.")
        video_url, query, ground_truths = task_gen.generate_task()

    print_header("Task Generation")
    print_task(task_number, video_url, query, ground_truths)

    request = verify_protocol_request(video_url, query)

    print_header("Miner Search")
    results = miner.search(request.video_url, request.query)
    response = verify_protocol_response(request.request_id, results)

    if not response.results:
        raise RuntimeError("Miner returned an empty result list.")

    if skip_quality_check:
        print_header("Pipeline Check")
        best = response.results[0]
        print(f"Top result: {best.start:.2f}s -> {best.end:.2f}s")
        print(f"Confidence: {best.confidence:.4f}")
        print("Quality check skipped.")
        return {
            "request": request,
            "response": response,
            "ground_truths": ground_truths,
            "raw_iou": None,
            "score": None,
        }

    print_header("Quality Check")
    best = response.results[0]
    raw_iou = best_iou([best], ground_truths)
    iou_score = score_response(response.results, ground_truths, latency=0.0)

    print(f"Top result: {best.start:.2f}s -> {best.end:.2f}s")
    print(f"Confidence: {best.confidence:.4f}")
    print(f"Raw IoU:    {raw_iou:.4f}")
    print(f"IoU Score:  {iou_score:.4f}")
    print(f"Strict Pass: {'yes' if passes_strict_iou(iou_score) else 'no'}")
    print("Top-result IoU by GT:")
    for start, end in ground_truths:
        iou = calculate_iou(best.start, best.end, start, end)
        print(f"  - {start:.2f}s -> {end:.2f}s: {iou:.4f}")

    return {
        "request": request,
        "response": response,
        "ground_truths": ground_truths,
        "raw_iou": raw_iou,
        "score": iou_score,
    }


def main():
    args = parse_args()

    bt.logging.set_info(True)
    print("ChronoSeek MVP Verification")
    print(f"Split: {args.split}")
    if args.video_url:
        print("Mode: custom video/query")
        print(f"Video URL: {args.video_url}")
        print(f"Query: {args.query}")
    else:
        print(f"Dataset path: {args.dataset_path}")

    task_gen = None
    if not args.video_url:
        try:
            task_gen = build_task_generator(args)
        except Exception as exc:
            print_header("Setup Failure")
            print(str(exc))
            return 1
    elif not args.query:
        print_header("Setup Failure")
        print("--query is required when --video-url is provided.")
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
            result = run_single_attempt(
                task_gen,
                miner,
                attempt_number,
                custom_video_url=args.video_url,
                custom_query=args.query,
                skip_quality_check=args.skip_quality_check or bool(args.video_url),
            )
            passed = (
                True
                if result["score"] is None
                else passes_strict_iou(result["score"])
            )

            print_header("Verification Summary")
            print("End-to-end flow: ok")
            print(
                "Strict IoU pass: "
                + ("skipped" if result["score"] is None else ("yes" if passed else "no"))
            )

            if args.require_pass and result["score"] is not None and not passed:
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
