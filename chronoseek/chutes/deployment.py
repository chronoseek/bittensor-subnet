"""Chutes API deployment helpers.

Chute and Image objects are still defined with the Chutes SDK, but build and
deploy are executed through Chutes HTTP APIs using `CHUTES_API_KEY`. This avoids
requiring miners to run `chutes login` or maintain local Chutes SDK configs.
"""

import base64
import importlib
import pickle
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import bittensor as bt
import httpx

from chronoseek.chutes.runtime import chutes_auth_headers_from_env


@dataclass(frozen=True)
class RuntimeMetadata:
    endpoint: str | None = None
    chute_id: str | None = None
    chute_slug: str | None = None
    artifact_id: str | None = None
    artifact_revision: str | None = None
    artifact_digest: str | None = None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value in (None, "") or isinstance(value, (dict, list, tuple)):
        return None
    stripped = str(value).strip()
    return stripped or None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text(value)
        if text:
            return text
    return None


def _image_id_from_response(image: Any) -> str | None:
    image_mapping = _mapping(image)
    if image_mapping:
        return _first_text(
            image_mapping.get("image_id"),
            image_mapping.get("id"),
            image_mapping.get("uid"),
            image_mapping.get("name"),
            image_mapping.get("repo"),
        )
    return _text(image)


def metadata_from_chutes_response(raw: dict[str, Any]) -> RuntimeMetadata:
    """Extract commit metadata from known Chutes API response fields."""

    if not isinstance(raw, dict):
        return RuntimeMetadata()

    chute_value = raw.get("chute")
    chute = _mapping(chute_value)
    artifact = _mapping(raw.get("artifact"))
    image_value = raw.get("image")
    image = _mapping(image_value)

    return RuntimeMetadata(
        endpoint=_first_text(
            raw.get("public_api"),
            raw.get("public_api_url"),
            raw.get("endpoint"),
            raw.get("url"),
            raw.get("base_url"),
            raw.get("runtime_url"),
            raw.get("private_url"),
            raw.get("public_url"),
            chute.get("public_api"),
            chute.get("public_api_url"),
            chute.get("endpoint"),
            chute.get("url"),
        ),
        chute_id=_first_text(
            raw.get("chute_id"),
            raw.get("id"),
            raw.get("uid"),
            chute.get("chute_id"),
            chute.get("id"),
            chute.get("uid"),
            None if chute else chute_value,
        ),
        chute_slug=_first_text(
            raw.get("chute_slug"),
            raw.get("slug"),
            raw.get("name"),
            chute.get("chute_slug"),
            chute.get("slug"),
            chute.get("name"),
        ),
        artifact_id=_first_text(
            raw.get("artifact_id"),
            raw.get("image_id"),
            raw.get("repo"),
            _image_id_from_response(image_value),
            artifact.get("artifact_id"),
            artifact.get("id"),
            artifact.get("uid"),
            artifact.get("name"),
            artifact.get("repo"),
        ),
        artifact_revision=_first_text(
            raw.get("artifact_revision"),
            raw.get("revision"),
            raw.get("revision_sha"),
            raw.get("sha"),
            raw.get("commit"),
            raw.get("commit_sha"),
            raw.get("tag"),
            artifact.get("artifact_revision"),
            artifact.get("revision"),
            artifact.get("revision_sha"),
            artifact.get("sha"),
            artifact.get("commit"),
            artifact.get("commit_sha"),
            artifact.get("tag"),
            image.get("tag"),
            image.get("revision"),
            image.get("revision_sha"),
        ),
        artifact_digest=_first_text(
            raw.get("artifact_digest"),
            raw.get("digest"),
            raw.get("image_digest"),
            artifact.get("artifact_digest"),
            artifact.get("digest"),
            artifact.get("image_digest"),
            image.get("digest"),
            image.get("image_digest"),
        ),
    )


def merge_metadata(
    *,
    from_args: RuntimeMetadata,
    from_api: RuntimeMetadata,
) -> RuntimeMetadata:
    return RuntimeMetadata(
        **{
            key: getattr(from_args, key) or getattr(from_api, key)
            for key in RuntimeMetadata.__annotations__
        }
    )


def normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    return str(value).rstrip("/")


def _image_metadata_from_definition(image: Any) -> tuple[str | None, str | None]:
    if image is None:
        return None, None
    if isinstance(image, str):
        return image, None
    image_id = getattr(image, "uid", None) or getattr(image, "name", None)
    image_tag = getattr(image, "tag", None)
    return (
        str(image_id) if image_id else None,
        str(image_tag) if image_tag else None,
    )


def image_id_from_definition(image: Any) -> str:
    if isinstance(image, str):
        raise ValueError(f"Image '{image}' is prebuilt; there is nothing to build.")
    image_id = getattr(image, "uid", None)
    if not image_id:
        raise ValueError("Chutes image definition does not include a usable image ID.")
    return str(image_id)


def image_label_from_definition(image: Any) -> str:
    username = getattr(image, "username", None)
    name = getattr(image, "name", None)
    tag = getattr(image, "tag", None)
    if username and name and tag:
        return f"{username}/{name}:{tag}"
    if name and tag:
        return f"{name}:{tag}"
    return image_id_from_definition(image)


def require_chute_module_ref(chute_ref: str) -> str:
    """Validate the module ref shape used for local chute definitions."""

    if "/" in chute_ref or chute_ref.endswith(".py") or ".py:" in chute_ref:
        raise ValueError(
            "Chutes deployment expects a module ref like "
            "`chronoseek_chute:chute`, not a file path. Copy the chute "
            "example to the repository root as `chronoseek_chute.py`."
        )
    if ":" not in chute_ref:
        raise ValueError("Chutes chute refs must use module:attribute format.")
    return chute_ref


def load_chute_definition(chute_ref: str):
    """Load a Chute object without requiring Chutes credentials/config."""

    require_chute_module_ref(chute_ref)
    from chutes.chute import Chute, ChutePack

    module_name, attr = chute_ref.split(":", 1)
    module = importlib.import_module(module_name)
    chute = getattr(module, attr)
    if isinstance(chute, ChutePack):
        chute = chute.chute
    if not isinstance(chute, Chute):
        raise TypeError(f"{chute_ref} did not resolve to a Chutes Chute object.")
    module_path = Path(module.__file__).resolve()
    return module, module_path, chute


def load_chute_object(chute_ref: str):
    """Load a Chute object using local Python imports only."""

    _, _, chute = load_chute_definition(chute_ref)
    return chute


def metadata_from_chute_object(chute: Any) -> RuntimeMetadata:
    image_id, image_tag = _image_metadata_from_definition(
        getattr(chute, "image", None)
    )
    revision = getattr(chute, "revision", None) or image_tag
    chute_name = getattr(chute, "name", None)
    username = getattr(chute, "username", None) or getattr(chute, "_username", None)
    chute_slug = getattr(chute, "slug", None)
    if not chute_slug and username and chute_name:
        chute_slug = f"{username}-{chute_name}"
    elif not chute_slug:
        chute_slug = chute_name

    return RuntimeMetadata(
        chute_id=str(getattr(chute, "uid", "")) or None,
        chute_slug=str(chute_slug) if chute_slug else None,
        artifact_id=image_id,
        artifact_revision=str(revision) if revision else None,
    )


def metadata_from_chute_definition(chute_ref: str) -> RuntimeMetadata:
    return metadata_from_chute_object(load_chute_object(chute_ref))


def _collect_build_context_paths(image: Any, *, include_cwd: bool) -> list[Path]:
    from chutes.image.directive.add import ADD

    if include_cwd:
        image._directives.append(ADD(source=".", dest="/app"))

    cwd = Path.cwd().resolve()
    paths: list[Path] = []
    for directive in getattr(image, "_directives", []):
        for raw_path in getattr(directive, "_build_context", []):
            path = Path(raw_path).expanduser().resolve()
            if path.is_dir():
                paths.extend(item for item in path.rglob("*") if item.is_file())
            elif path.is_file():
                paths.append(path)

    filtered: list[Path] = []
    for path in paths:
        if path.name == "Dockerfile":
            continue
        try:
            path.relative_to(cwd)
        except ValueError:
            continue
        filtered.append(path)
    return sorted(set(filtered))


def build_context_zip_bytes(image: Any, *, include_cwd: bool = False) -> bytes:
    cwd = Path.cwd().resolve()
    paths = _collect_build_context_paths(image, include_cwd=include_cwd)
    bt.logging.info(f"Packaging {len(paths)} Chutes image build context files.")

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(cwd))
    return buffer.getvalue()


def image_build_form_payload(
    image: Any,
    *,
    include_cwd: bool = False,
    public: bool = False,
    logo_id: str | None = None,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    if isinstance(image, str):
        raise ValueError(f"Image '{image}' is prebuilt; there is nothing to build.")

    data = {
        "username": str(image.username),
        "name": str(image.name),
        "tag": str(image.tag),
        "readme": str(getattr(image, "readme", "") or ""),
        "dockerfile": str(image),
        "public": str(bool(public)),
        "logo_id": str(logo_id) if logo_id else "__none__",
        "wait": "False",
        "image": base64.b64encode(pickle.dumps(image)).decode(),
    }
    files = {
        "build_context": (
            "chute.zip",
            build_context_zip_bytes(image, include_cwd=include_cwd),
            "application/zip",
        )
    }
    return data, files


def _raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()
        if detail:
            raise RuntimeError(
                f"{exc}. Chutes response body: {detail[:2000]}"
            ) from exc
        raise


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


async def get_image_via_api(
    *,
    api_base_url: str,
    image_id: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any] | None:
    """Return Chutes image metadata, or None when the image does not exist."""

    url = f"{api_base_url.rstrip('/')}/images/{image_id}"
    bt.logging.info(f"Checking Chutes image existence through API: GET {url}")
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        response = await client.get(
            url,
            headers=chutes_auth_headers_from_env(require_token=True),
        )
        if response.status_code == 404:
            return None
        _raise_for_status_with_body(response)
        return _json_or_empty(response)


async def delete_image_via_api(
    *,
    api_base_url: str,
    image_id: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Delete a Chutes image by ID or name."""

    url = f"{api_base_url.rstrip('/')}/images/{image_id}"
    bt.logging.warning(f"Deleting existing Chutes image through API: DELETE {url}")
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        response = await client.delete(
            url,
            headers=chutes_auth_headers_from_env(require_token=True),
        )
        _raise_for_status_with_body(response)
        return _json_or_empty(response)


def confirm_image_overwrite(*, image_id: str, image_label: str) -> bool:
    print(
        f"Chutes image already exists: {image_label} ({image_id}).",
        flush=True,
    )
    try:
        answer = input("Delete it and build a new image? [Y/n]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"", "y", "yes"}


async def submit_image_build_via_api(
    *,
    api_base_url: str,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/images/"
    bt.logging.info(f"Starting Chutes image build through API: POST {url}")
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        response = await client.post(
            url,
            data=data,
            files=files,
            headers=chutes_auth_headers_from_env(require_token=True),
        )
        _raise_for_status_with_body(response)
        return response.json()


def chute_deploy_payload(
    *,
    chute_ref: str,
    public: bool = False,
    logo_id: str | None = None,
) -> dict[str, Any]:
    _, module_path, chute = load_chute_definition(chute_ref)
    image = chute.image if isinstance(chute.image, str) else chute.image.uid
    node_selector = (
        chute.node_selector.model_dump()
        if hasattr(chute.node_selector, "model_dump")
        else chute.node_selector.dict()
    )
    return {
        "name": chute.name,
        "tagline": chute.tagline,
        "readme": chute.readme,
        "logo_id": logo_id,
        "image": image,
        "public": bool(public),
        "standard_template": chute.standard_template,
        "node_selector": node_selector,
        "filename": module_path.name,
        "ref_str": chute_ref,
        "code": module_path.read_text(),
        "concurrency": chute.concurrency,
        "max_instances": chute.max_instances,
        "scaling_threshold": chute.scaling_threshold,
        "shutdown_after_seconds": chute.shutdown_after_seconds,
        "allow_external_egress": chute.allow_external_egress,
        "encrypted_fs": chute.encrypted_fs,
        "tee": chute.tee,
        "lock_modules": chute.lock_modules,
        "revision": chute.revision,
        "cords": [
            {
                "method": cord._method,
                "path": cord.path,
                "public_api_path": cord.public_api_path,
                "public_api_method": cord._public_api_method,
                "stream": cord._stream,
                "function": cord._func.__name__,
                "input_schema": cord.input_schema,
                "output_schema": cord.output_schema,
                "output_content_type": cord.output_content_type,
                "minimal_input_schema": cord.minimal_input_schema,
                "passthrough": cord._passthrough,
            }
            for cord in chute._cords
        ],
        "jobs": [
            {
                "ports": [
                    {
                        "name": port.name,
                        "port": port.port,
                        "proto": port.proto,
                    }
                    for port in job.ports
                ],
                "timeout": job.timeout,
                "name": job._name,
                "upload": job.upload,
            }
            for job in chute._jobs
        ],
    }


async def build_image_via_api(
    *,
    api_base_url: str,
    chute_ref: str,
    include_cwd: bool = False,
    public: bool = False,
    overwrite_existing: bool | None = None,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Create a Chutes image build through `POST /images/`."""

    image = load_chute_object(chute_ref).image
    image_id = image_id_from_definition(image)
    image_label = image_label_from_definition(image)
    existing_image = await get_image_via_api(
        api_base_url=api_base_url,
        image_id=image_id,
        timeout_seconds=timeout_seconds,
    )
    if existing_image is not None:
        if overwrite_existing is None:
            overwrite_existing = confirm_image_overwrite(
                image_id=image_id,
                image_label=image_label,
            )
        else:
            bt.logging.warning(
                f"Chutes image already exists: {image_label} ({image_id})."
            )
        if not overwrite_existing:
            raise RuntimeError(
                "Chutes image build aborted because the image already exists. "
                "Use --overwrite-existing-image to delete and rebuild it."
            )
        await delete_image_via_api(
            api_base_url=api_base_url,
            image_id=image_id,
            timeout_seconds=timeout_seconds,
        )

    data, files = image_build_form_payload(
        image,
        include_cwd=include_cwd,
        public=public,
    )
    return await submit_image_build_via_api(
        api_base_url=api_base_url,
        data=data,
        files=files,
        timeout_seconds=timeout_seconds,
    )


async def stream_image_build_logs_via_api(
    *,
    api_base_url: str,
    image_id: str,
    timeout_seconds: float = 3600.0,
) -> None:
    """Stream Chutes image build logs through `GET /images/{image_id}/logs`."""

    import orjson as json

    def _build_failed(log_line: str) -> bool:
        text = str(log_line).strip().lower()
        if not text:
            return False
        error_markers = (
            "permission denied",
            "exit status",
            "subprocess exited with status",
            "error: building at step",
            "failed to solve",
            "traceback (most recent call last)",
            "exception:",
        )
        return any(marker in text for marker in error_markers)

    params: dict[str, str] = {}
    url = f"{api_base_url.rstrip('/')}/images/{image_id}/logs"
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        async with client.stream(
            "GET",
            url,
            params=params,
            headers=chutes_auth_headers_from_env(require_token=True),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/plain" in content_type:
                text = await response.aread()
                decoded = text.decode(errors="replace").strip()
                if decoded:
                    print(decoded)
                    if _build_failed(decoded):
                        raise RuntimeError(
                            "Chutes image build failed during log streaming. "
                            f"Last log output:\n{decoded[-2000:]}"
                        )
                return

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("DONE"):
                    return
                if not line.startswith("data: {"):
                    continue
                try:
                    payload = json.loads(line[6:])
                except Exception:
                    continue
                if payload.get("offset") is not None:
                    params["offset"] = str(payload["offset"])
                log_data = payload.get("log")
                if isinstance(log_data, dict):
                    log_text = str(log_data.get("log", "")).strip()
                else:
                    log_text = str(log_data or "").strip()
                if log_text:
                    print(log_text)
                    if _build_failed(log_text):
                        raise RuntimeError(
                            "Chutes image build failed during log streaming. "
                            f"Failing log line: {log_text}"
                        )


async def deploy_chute_via_api(
    *,
    api_base_url: str,
    chute_ref: str,
    accept_fee: bool = False,
    public: bool = False,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Deploy a Chute through `POST /chutes/`."""

    payload = chute_deploy_payload(chute_ref=chute_ref, public=public)
    url = f"{api_base_url.rstrip('/')}/chutes/"
    bt.logging.info(f"Deploying Chutes runtime through API: POST {url}")
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        response = await client.post(
            url,
            params={"accept_fee": str(bool(accept_fee)).lower()},
            json=payload,
            headers=chutes_auth_headers_from_env(
                require_token=True,
                include_content_type=True,
            ),
        )
        _raise_for_status_with_body(response)
        return response.json()


async def get_chute_via_api(
    *,
    api_base_url: str,
    chute_id_or_name: str,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Load Chutes deployment metadata through `GET /chutes/{id_or_name}`."""

    url = f"{api_base_url.rstrip('/')}/chutes/{chute_id_or_name}"
    bt.logging.info(f"Loading Chutes runtime metadata through API: GET {url}")
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        response = await client.get(
            url,
            headers=chutes_auth_headers_from_env(require_token=True),
        )
        _raise_for_status_with_body(response)
        return response.json()
