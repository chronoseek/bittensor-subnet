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
from datetime import UTC, datetime
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
    resolve_chute_api_name,
    resolve_chute_api_runtime_name,
    resolve_chute_display_name,
    resolve_chute_logo_url,
    resolve_chute_slug,
    load_chute_object,
    require_chute_module_ref,
    stream_image_build_logs_via_api,
    upload_logo_via_api,
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
        help="Explicit deployed runtime endpoint to include in deployment metadata.",
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


def resolve_runtime_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")[:-3]


def chute_username(chute) -> str:
    return str(
        getattr(chute, "username", None) or getattr(chute, "_username", None) or ""
    ).strip()


def chute_logo_url(chute) -> str:
    return resolve_chute_logo_url(chute)


def explicit_metadata(config) -> RuntimeMetadata:
    return RuntimeMetadata(
        endpoint=normalize_url(config.endpoint) or None,
        chute_id=config.chute_id or None,
        chute_slug=config.chute_slug or None,
        artifact_id=config.artifact_id or None,
        artifact_revision=config.artifact_revision or None,
        artifact_digest=config.artifact_digest or None,
    )


def metadata_with_chute_slug(
    metadata: RuntimeMetadata,
    chute_slug: str,
) -> RuntimeMetadata:
    return RuntimeMetadata(
        endpoint=metadata.endpoint,
        chute_id=metadata.chute_id,
        chute_slug=chute_slug or metadata.chute_slug,
        artifact_id=metadata.artifact_id,
        artifact_revision=metadata.artifact_revision,
        artifact_digest=metadata.artifact_digest,
    )


def miner_command(metadata: RuntimeMetadata, config) -> list[str]:
    command = [
        "poetry",
        "run",
        "python",
        "miner.py",
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

    print("\nCommit this deployment on-chain with miner.py:")
    print(shlex.join(miner_command(metadata, config)))


async def run(config) -> tuple[RuntimeMetadata, dict[str, object]]:
    require_chute_module_ref(config.chute_ref)
    runtime_timestamp = resolve_runtime_timestamp()
    explicit = explicit_metadata(config)
    base_chute = load_chute_object(config.chute_ref)
    logo_url = chute_logo_url(base_chute)
    base_chute_name = getattr(base_chute, "name", None) or "chronoseek-runtime"
    chute_base_name = resolve_chute_api_name(base_chute_name)
    chute_name = resolve_chute_api_runtime_name(chute_base_name, runtime_timestamp)
    chute_display_name = resolve_chute_display_name(chute_base_name)
    chute_slug = resolve_chute_slug(
        chute_username(base_chute),
        chute_base_name,
        runtime_timestamp,
    )
    bt.logging.info(f"Resolved Chutes runtime timestamp: {runtime_timestamp}")
    bt.logging.info(f"Resolved Chutes display label: {chute_display_name}")
    bt.logging.info(f"Resolved Chutes API name: {chute_name}")
    bt.logging.info(f"Resolved Chutes slug: {chute_slug}")
    bt.logging.info(f"Resolved Chutes logo URL: {logo_url}")
    definition_metadata = metadata_from_chute_definition(
        config.chute_ref,
        chute_name=chute_name,
        chute_slug=chute_slug,
        chute_display_name=chute_display_name,
    )
    api_metadata = RuntimeMetadata()
    raw_responses: dict[str, object] = {}
    logo_id: str | None = None

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
        if config.build or config.deploy:
            logo_id = await upload_logo_via_api(
                api_base_url=config.chutes_api_base_url,
                logo_url=logo_url,
                timeout_seconds=min(float(config.chutes_timeout_seconds), 120.0),
            )
            raw_responses["logo"] = {"logo_id": logo_id, "url": logo_url}
            bt.logging.info(f"Resolved Chutes logo ID: {logo_id}")

        if config.build:
            raw_responses["image"] = await build_image_via_api(
                api_base_url=config.chutes_api_base_url,
                chute_ref=config.chute_ref,
                include_cwd=bool(config.include_cwd),
                public=bool(config.public),
                overwrite_existing=config.overwrite_existing_image,
                timeout_seconds=float(config.chutes_timeout_seconds),
                chute_name=chute_name,
                chute_slug=chute_slug,
                chute_display_name=chute_display_name,
                logo_id=logo_id,
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
                chute_name=chute_name,
                chute_slug=chute_slug,
                chute_display_name=chute_display_name,
                logo_id=logo_id,
            )
            api_metadata = metadata_with_chute_slug(
                metadata_from_chutes_response(raw_responses["chute"]),
                chute_slug,
            )

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
