#!/usr/bin/env python3
"""Build/deploy/warmup a ChronoSeek miner runtime through production Chutes APIs.

The Chute and Image are defined with the Chutes SDK in `chronoseek_chute.py`,
but this script performs the actual image build and chute deployment through
Chutes HTTP APIs using `CHUTES_API_KEY`. Miners do not need `chutes login` or a
local Chutes SDK config file. Use `scripts/test_chutes_runtime_local.py` for
local, no-credit testing before running this helper.
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

from dotenv import load_dotenv

from chronoseek.chain.submissions import MinerSubmission
from chronoseek.chutes.deployment import (
    ChutesWarmupInterrupted,
    ChutesWarmupTimeout,
    RuntimeMetadata,
    build_image_via_api,
    deploy_chute_via_api,
    get_chutes_username_via_api,
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
    warmup_chute_via_api,
)
from chronoseek.chutes.runtime import resolve_submission_endpoint
from chronoseek.constants import (
    DEFAULT_CHUTES_API_BASE_URL,
    DEFAULT_CHUTES_BASE_DOMAIN,
    DEFAULT_LOG_LEVEL,
)
from chronoseek.logging import configure_logging as configure_application_logging
from chronoseek.logging import logger

sys.argv = _ORIGINAL_ARGV
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/deploy a ChronoSeek Chutes runtime through production "
            "Chutes APIs and print the metadata needed for miner.py."
        )
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--chutes-api-base-url",
        type=str,
        default=DEFAULT_CHUTES_API_BASE_URL,
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
        "--warmup",
        action="store_true",
        default=False,
        help="After a successful --deploy, warm up the chute until Chutes reports it hot.",
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
            "Chutes SDK definition; chronoseek_chute.py derives its image/chute "
            "revision from the current git commit."
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
        default=3600.0,
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
    configure_application_logging(config.log_level)


def resolve_runtime_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")[:-3]


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
        chutes_base_domain=DEFAULT_CHUTES_BASE_DOMAIN,
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
        logger.success(f"Wrote deployment metadata to {output_path}")

    print("\nCommit this deployment on-chain with miner.py:")
    print(shlex.join(miner_command(metadata, config)))


def manual_warmup_command(chute_id_or_name: str) -> list[str]:
    return ["chutes", "warmup", chute_id_or_name]


def manual_warmup_api_url(api_base_url: str, chute_id_or_name: str) -> str:
    return f"{api_base_url.rstrip('/')}/chutes/warmup/{chute_id_or_name}?quick=true"


def print_warmup_followup(
    *,
    api_base_url: str,
    chute_id_or_name: str,
    reason: str,
) -> None:
    command = shlex.join(manual_warmup_command(chute_id_or_name))
    api_url = manual_warmup_api_url(api_base_url, chute_id_or_name)
    message = (
        "\nChutes deployment completed successfully, but this helper did not "
        "finish warming the runtime to hot.\n"
        "The deployment metadata is still valid. Validators may not be able "
        "to call the runtime until Chutes reports it hot.\n\n"
        f"Reason: {reason}\n\n"
        "Continue warmup manually with:\n"
        f"  {command}\n\n"
        "Or call the Chutes API:\n"
        f"  GET {api_url}\n"
    )
    print(message)
    logger.warning(
        "Chutes deployment succeeded, but warmup is still pending. "
        f"Run `{command}` or call GET {api_url}"
    )


async def run(config) -> tuple[RuntimeMetadata, dict[str, object]]:
    require_chute_module_ref(config.chute_ref)
    if config.warmup and not config.deploy:
        raise RuntimeError("--warmup requires --deploy.")

    runtime_timestamp = resolve_runtime_timestamp()
    explicit = explicit_metadata(config)
    base_chute = load_chute_object(config.chute_ref)
    logo_url = chute_logo_url(base_chute)
    chutes_username = await get_chutes_username_via_api(
        api_base_url=config.chutes_api_base_url,
        timeout_seconds=float(config.chutes_timeout_seconds),
    )
    base_chute_name = getattr(base_chute, "name", None) or "chronoseek-runtime"
    chute_base_name = resolve_chute_api_name(base_chute_name)
    chute_name = resolve_chute_api_runtime_name(chute_base_name, runtime_timestamp)
    chute_display_name = resolve_chute_display_name(chute_base_name)
    chute_slug = resolve_chute_slug(
        chutes_username,
        chute_base_name,
        runtime_timestamp,
    )
    logger.info(f"Resolved Chutes username: {chutes_username}")
    logger.info(f"Resolved Chutes runtime timestamp: {runtime_timestamp}")
    logger.info(f"Resolved Chutes display label: {chute_display_name}")
    logger.info(f"Resolved Chutes API name: {chute_name}")
    logger.info(f"Resolved Chutes slug: {chute_slug}")
    logger.info(f"Resolved Chutes logo URL: {logo_url}")
    definition_metadata = metadata_from_chute_definition(
        config.chute_ref,
        chutes_username=chutes_username,
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
            try:
                logo_id = await upload_logo_via_api(
                    api_base_url=config.chutes_api_base_url,
                    logo_url=logo_url,
                    timeout_seconds=min(float(config.chutes_timeout_seconds), 120.0),
                )
                raw_responses["logo"] = {"logo_id": logo_id, "url": logo_url}
                logger.info(f"Resolved Chutes logo ID: {logo_id}")
            except Exception as exc:
                logger.warning(
                    f"Logo upload failed ({exc}); continuing deploy without a logo."
                )
                logo_id = None

        if config.build:
            raw_responses["image"] = await build_image_via_api(
                api_base_url=config.chutes_api_base_url,
                chute_ref=config.chute_ref,
                include_cwd=bool(config.include_cwd),
                public=bool(config.public),
                overwrite_existing=config.overwrite_existing_image,
                timeout_seconds=float(config.chutes_timeout_seconds),
                chutes_username=chutes_username,
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
                chutes_username=chutes_username,
                chute_name=chute_name,
                chute_slug=chute_slug,
                chute_display_name=chute_display_name,
                logo_id=logo_id,
            )
            api_metadata = metadata_with_chute_slug(
                metadata_from_chutes_response(raw_responses["chute"]),
                chute_slug,
            )
            logger.success("Chutes runtime deployment completed successfully.")
            if config.warmup:
                warmup_target = (
                    api_metadata.chute_id or api_metadata.chute_slug or chute_name
                )
                try:
                    raw_responses["warmup"] = await warmup_chute_via_api(
                        api_base_url=config.chutes_api_base_url,
                        chute_id_or_name=warmup_target,
                        timeout_seconds=float(config.chutes_timeout_seconds),
                    )
                except ChutesWarmupInterrupted as exc:
                    command = shlex.join(manual_warmup_command(warmup_target))
                    api_url = manual_warmup_api_url(
                        config.chutes_api_base_url,
                        warmup_target,
                    )
                    raw_responses["warmup"] = {
                        "status": "interrupted",
                        "target": warmup_target,
                        "last_response": exc.last_payload,
                        "manual_command": command,
                        "api_url": api_url,
                    }
                    print_warmup_followup(
                        api_base_url=config.chutes_api_base_url,
                        chute_id_or_name=warmup_target,
                        reason="warmup was interrupted before the chute became hot",
                    )
                except ChutesWarmupTimeout as exc:
                    command = shlex.join(manual_warmup_command(warmup_target))
                    api_url = manual_warmup_api_url(
                        config.chutes_api_base_url,
                        warmup_target,
                    )
                    raw_responses["warmup"] = {
                        "status": "not_hot_yet",
                        "target": warmup_target,
                        "timeout_seconds": exc.timeout_seconds,
                        "last_response": exc.last_payload,
                        "manual_command": command,
                        "api_url": api_url,
                    }
                    print_warmup_followup(
                        api_base_url=config.chutes_api_base_url,
                        chute_id_or_name=warmup_target,
                        reason=(
                            f"warmup did not become hot within "
                            f"{exc.timeout_seconds:g} seconds"
                        ),
                    )
                except Exception as exc:
                    command = shlex.join(manual_warmup_command(warmup_target))
                    api_url = manual_warmup_api_url(
                        config.chutes_api_base_url,
                        warmup_target,
                    )
                    raw_responses["warmup"] = {
                        "status": "failed",
                        "target": warmup_target,
                        "error": str(exc),
                        "manual_command": command,
                        "api_url": api_url,
                    }
                    print_warmup_followup(
                        api_base_url=config.chutes_api_base_url,
                        chute_id_or_name=warmup_target,
                        reason=str(exc),
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
        logger.success("Chutes API runtime metadata is ready for miner.py.")
        return 0
    except Exception as exc:
        logger.error(f"Chutes API runtime deployment helper failed: {exc}")
        return 1


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
