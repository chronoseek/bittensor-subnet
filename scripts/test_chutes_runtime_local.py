#!/usr/bin/env python3
"""Build, run, or smoke-test the ChronoSeek Chutes runtime locally.

This helper follows the Chutes local testing flow:

    chutes build chronoseek_chute_local:chute --local
    docker run ... chutes run chronoseek_chute_local:chute --dev

It never calls the production Chutes API.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from chronoseek.config import PROTOCOL_VERSION

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only ChronoSeek Chutes runtime test helper."
    )
    parser.add_argument(
        "--chute-ref",
        default="chronoseek_chute_local:chute",
        help="Chutes SDK module ref to test locally.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--build", action="store_true", help="Run chutes build --local.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the local Docker container. This blocks until stopped.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test an already-running local runtime over HTTP.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print local build/run commands without executing them.",
    )
    parser.add_argument(
        "--query",
        default="a rabbit standing outside",
        help="Query used for the optional /search smoke test.",
    )
    parser.add_argument(
        "--video-url",
        default="https://www.w3schools.com/html/mov_bbb.mp4",
        help="Video URL used for the optional /search smoke test.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only call /health during smoke testing.",
    )
    return parser


def load_image_ref(chute_ref: str) -> str:
    from chronoseek.chutes.deployment import load_chute_object, require_chute_module_ref

    require_chute_module_ref(chute_ref)
    chute = load_chute_object(chute_ref)
    image = getattr(chute, "image", None)
    name = str(getattr(image, "name", "") or "").strip()
    tag = str(getattr(image, "tag", "") or "").strip()
    if not name or not tag:
        raise RuntimeError("Chute image must define both image.name and image.tag.")
    return f"{name}:{tag}"


def local_build_command(chute_ref: str) -> list[str]:
    chutes_bin = shutil.which("chutes") or "chutes"
    return [chutes_bin, "build", chute_ref, "--local"]


def local_run_command(chute_ref: str, image_ref: str, *, port: int, env_file: str) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "-it",
    ]
    env_path = Path(env_file).expanduser()
    if env_path.is_file():
        command.extend(["--env-file", str(env_path)])
    command.extend(
        [
            "-e",
            "CHUTES_EXECUTION_CONTEXT=REMOTE",
            "-p",
            f"{port}:{port}",
            image_ref,
            "chutes",
            "run",
            chute_ref,
            "--port",
            str(port),
            "--dev",
        ]
    )
    return command


def print_commands(build_cmd: list[str], run_cmd: list[str]) -> None:
    print("Local Chutes build command:")
    print(shlex.join(build_cmd))
    print("\nLocal Chutes run command:")
    print(shlex.join(run_cmd))


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def smoke_test(config) -> None:
    base_url = f"http://127.0.0.1:{config.port}"
    with httpx.Client(timeout=float(config.timeout_seconds)) as client:
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        print("Local /health response:")
        print(json.dumps(health.json(), indent=2, sort_keys=True))

        if config.health_only:
            return

        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "local-chutes-smoke-test",
            "query": config.query,
            "top_k": int(config.top_k),
            "video": {"url": config.video_url},
        }
        search = client.post(f"{base_url}/search", json=payload)
        search.raise_for_status()
        print("\nLocal /search response:")
        print(json.dumps(search.json(), indent=2, sort_keys=True))


def main() -> int:
    config = build_parser().parse_args()
    image_ref = load_image_ref(config.chute_ref)
    build_cmd = local_build_command(config.chute_ref)
    run_cmd = local_run_command(
        config.chute_ref,
        image_ref,
        port=int(config.port),
        env_file=config.env_file,
    )

    if config.print_commands or not (config.build or config.run or config.smoke):
        print_commands(build_cmd, run_cmd)

    try:
        if config.build:
            run_command(build_cmd)
        if config.run:
            run_command(run_cmd)
        if config.smoke:
            smoke_test(config)
    except Exception as exc:
        print(f"Local Chutes runtime test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
