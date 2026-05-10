#!/usr/bin/env python3
"""Build/deploy a ChronoSeek miner runtime through Chutes APIs.

The Chute and Image are defined with the Chutes SDK in `chronoseek_chute.py`,
but this script performs the actual image build and chute deployment through
Chutes HTTP APIs using `CHUTES_API_KEY`. Miners do not need `chutes login` or a
local Chutes SDK config file.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path

# Bittensor parses --help during import in some versions. Preserve normal
# argparse help for this wrapper by hiding help flags until imports complete.
_ORIGINAL_ARGV = sys.argv[:]
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    sys.argv = [sys.argv[0]]

import bittensor as bt
from dotenv import load_dotenv

from chronoseek.chain.submissions import MinerSubmission
from chronoseek.chutes.deployment import (
    RuntimeMetadata,
    build_image_via_api,
    deploy_chute_via_api,
    get_chute_via_api,
    merge_metadata,
    metadata_from_chute_definition,
    metadata_from_chutes_response,
    normalize_url,
    require_chute_module_ref,
    stream_image_build_logs_via_api,
)
from chronoseek.chutes.runtime import resolve_submission_endpoint

sys.argv = _ORIGINAL_ARGV
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/deploy a ChronoSeek Chutes runtime through Chutes APIs "
            "and print the metadata needed for miner.py."
        )
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--chutes-api-base-url",
        type=str,
        default="https://api.chutes.ai",
        help="Chutes API base URL.",
    )
    parser.add_argument(
        "--chute-ref",
        type=str,
        default="chronoseek_chute:chute",
        help=(
            "Chutes SDK module ref, e.g. chronoseek_chute:chute. The API "
            "payload is generated from this SDK object."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Create the Chutes image build through POST /images/.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the Chute through POST /chutes/.",
    )
    parser.add_argument(
        "--lookup-only",
        action="store_true",
        help="Do not build/deploy. Load metadata for --chute-id or --chute-slug.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream image build logs after POST /images/ when --build is used.",
    )
    parser.add_argument(
        "--include-cwd",
        action="store_true",
        help="Include the whole current working directory in the Chutes image build context.",
    )
    parser.add_argument(
        "--accept-fee",
        action="store_true",
        default=False,
        help="Pass accept_fee=true to Chutes deployment.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Mark image/deployment public when supported by the Chutes account.",
    )
    parser.add_argument(
        "--overwrite-existing-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When --build finds an existing image with the same generated image "
            "ID, delete it before building. If omitted, the wrapper prompts."
        ),
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="",
        help="Explicit deployed runtime endpoint to commit on-chain.",
    )
    parser.add_argument("--chute-id", type=str, default="")
    parser.add_argument("--chute-slug", type=str, default="")
    parser.add_argument(
        "--artifact-id",
        dest="artifact_id",
        type=str,
        default="",
        help="Optional runtime artifact identifier, such as a Chutes image name.",
    )
    parser.add_argument(
        "--artifact-revision",
        "--revision",
        dest="artifact_revision",
        type=str,
        default="",
        help=(
            "Optional on-chain provenance revision. This does not mutate the "
            "Chutes SDK definition; copy chronoseek_chute.example.py to "
            "chronoseek_chute.py and set RUNTIME_REVISION there for the actual "
            "image/chute revision."
        ),
    )
    parser.add_argument("--artifact-digest", type=str, default="")
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Runtime capability to include in the suggested miner.py command. Can be repeated.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=int(os.getenv("NETUID", "1")),
        help="Subnet NetUID used in the suggested miner.py command.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=os.getenv("NETWORK", "finney"),
        help="Bittensor network used in the suggested miner.py command.",
    )
    parser.add_argument(
        "--chutes-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for Chutes API build/deploy requests.",
    )
    parser.add_argument(
        "--output-metadata-path",
        type=str,
        default="",
        help="Optional path to write normalized deployment metadata as JSON.",
    )
    parser.add_argument(
        "--print-raw-response",
        action="store_true",
        help="Print raw Chutes API responses.",
    )
    return parser


def configure_logging(config) -> None:
    bt.logging.on()
    level = str(config.log_level).upper()
    if level == "DEBUG":
        bt.logging.set_debug(True)
    elif level == "TRACE":
        bt.logging.set_trace(True)
    else:
        bt.logging.set_info(True)


def explicit_metadata(config) -> RuntimeMetadata:
    return RuntimeMetadata(
        endpoint=normalize_url(config.endpoint) or None,
        chute_id=config.chute_id or None,
        chute_slug=config.chute_slug or None,
        artifact_id=config.artifact_id or None,
        artifact_revision=config.artifact_revision or None,
        artifact_digest=config.artifact_digest or None,
    )


def miner_command(metadata: RuntimeMetadata, config) -> list[str]:
    command = [
        "poetry",
        "run",
        "python",
        "miner.py",
        "--subtensor.network",
        str(config.network),
        "--netuid",
        str(config.netuid),
    ]
    field_to_flag = {
        "endpoint": "--endpoint",
        "chute_id": "--chute-id",
        "chute_slug": "--chute-slug",
        "artifact_id": "--artifact-id",
        "artifact_revision": "--artifact-revision",
        "artifact_digest": "--artifact-digest",
    }
    for field, flag in field_to_flag.items():
        value = getattr(metadata, field)
        if value:
            command.extend([flag, str(value)])
    for capability in config.capability or []:
        command.extend(["--capability", str(capability)])
    return command


def resolved_runtime_endpoint(metadata: RuntimeMetadata) -> str | None:
    submission = MinerSubmission(
        endpoint=metadata.endpoint or None,
        chute_id=metadata.chute_id or None,
        chute_slug=metadata.chute_slug or None,
    )
    return resolve_submission_endpoint(
        submission,
        chutes_base_domain=os.getenv("CHUTES_BASE_DOMAIN", "chutes.ai"),
    )


def print_results(metadata: RuntimeMetadata, config) -> None:
    payload = {
        key: value
        for key, value in {
            "endpoint": metadata.endpoint,
            "chute_id": metadata.chute_id,
            "chute_slug": metadata.chute_slug,
            "artifact_id": metadata.artifact_id,
            "artifact_revision": metadata.artifact_revision,
            "artifact_digest": metadata.artifact_digest,
        }.items()
        if value
    }
    print("\nNormalized deployment metadata:")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if config.output_metadata_path:
        output_path = Path(config.output_metadata_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        bt.logging.success(f"Wrote deployment metadata to {output_path}")

    print("\nCommit this deployment on-chain with:")
    print(shlex.join(miner_command(metadata, config)))


async def run(config) -> tuple[RuntimeMetadata, dict[str, object]]:
    require_chute_module_ref(config.chute_ref)
    explicit = explicit_metadata(config)
    definition_metadata = metadata_from_chute_definition(config.chute_ref)
    api_metadata = RuntimeMetadata()
    raw_responses: dict[str, object] = {}

    if config.lookup_only:
        chute_id_or_name = config.chute_id or config.chute_slug
        if not chute_id_or_name:
            raise RuntimeError("--lookup-only requires --chute-id or --chute-slug")
        raw_responses["lookup"] = await get_chute_via_api(
            api_base_url=config.chutes_api_base_url,
            chute_id_or_name=chute_id_or_name,
            timeout_seconds=float(config.chutes_timeout_seconds),
        )
        api_metadata = metadata_from_chutes_response(raw_responses["lookup"])
    else:
        if config.build:
            raw_responses["image"] = await build_image_via_api(
                api_base_url=config.chutes_api_base_url,
                chute_ref=config.chute_ref,
                include_cwd=bool(config.include_cwd),
                public=bool(config.public),
                overwrite_existing=config.overwrite_existing_image,
                timeout_seconds=float(config.chutes_timeout_seconds),
            )
            image_id = (
                raw_responses["image"].get("image_id")
                if isinstance(raw_responses["image"], dict)
                else None
            )
            if config.wait and image_id:
                await stream_image_build_logs_via_api(
                    api_base_url=config.chutes_api_base_url,
                    image_id=str(image_id),
                    timeout_seconds=float(config.chutes_timeout_seconds),
                )

        if config.deploy:
            raw_responses["chute"] = await deploy_chute_via_api(
                api_base_url=config.chutes_api_base_url,
                chute_ref=config.chute_ref,
                accept_fee=bool(config.accept_fee),
                public=bool(config.public),
                timeout_seconds=float(config.chutes_timeout_seconds),
            )
            api_metadata = metadata_from_chutes_response(raw_responses["chute"])

    return (
        merge_metadata(
            from_args=explicit,
            from_api=merge_metadata(
                from_args=api_metadata,
                from_api=definition_metadata,
            ),
        ),
        raw_responses,
    )


async def main_async() -> int:
    parser = build_parser()
    config = parser.parse_args()
    configure_logging(config)

    try:
        metadata, raw_responses = await run(config)
        if config.print_raw_response:
            print("\nRaw Chutes responses:")
            print(json.dumps(raw_responses, indent=2, sort_keys=True))

        if not resolved_runtime_endpoint(metadata):
            raise RuntimeError(
                "deployment metadata does not include endpoint or chute_slug; "
                "validators cannot route to this runtime yet"
            )

        print_results(metadata, config)
        bt.logging.success("Chutes API runtime metadata is ready for miner.py.")
        return 0
    except Exception as exc:
        bt.logging.error(f"Chutes API runtime deployment helper failed: {exc}")
        return 1


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
