import asyncio

import chronoseek.chutes.deployment as chutes_deployment
from chronoseek.chain.submissions import MinerSubmission
from chronoseek.chutes.deployment import (
    RuntimeMetadata,
    apply_runtime_name,
    build_image_via_api,
    chute_deploy_payload,
    image_build_form_payload,
    merge_metadata,
    metadata_from_chute_object,
    metadata_from_chutes_response,
    require_chute_module_ref,
    resolve_chute_api_name,
    resolve_chute_api_runtime_name,
    resolve_chute_display_name,
    resolve_chute_logo_url,
    resolve_chute_slug,
    upload_logo_via_api,
)
from chronoseek.chutes.runtime import resolve_submission_endpoint
import scripts.deploy_chutes_runtime as deploy_chutes_runtime
from scripts.deploy_chutes_runtime import (
    explicit_metadata,
    metadata_with_chute_slug,
    miner_command,
)


class DummyConfig:
    endpoint = "https://explicit.example.com/"
    chute_id = "chute-123"
    chute_slug = "chronoseek-runtime"
    artifact_id = "chronoseek/runtime"
    artifact_revision = "abc123"
    artifact_digest = "sha256:deadbeef"
    capability = ["vision", "audio"]


class DummyChutesImage:
    username = "chronoseek"
    name = "runtime"
    tag = "rev-1"
    readme = "Runtime readme."
    uid = "image-123"
    _directives = []

    def __str__(self):
        return "FROM parachutes/python:3.12"


class DummyChutesChute:
    image = DummyChutesImage()


def test_metadata_from_chutes_api_response():
    metadata = metadata_from_chutes_response(
        {
            "public_api": "https://runtime.example.com",
            "image": "chronoseek/runtime",
            "chute": {
                "id": "chute-123",
                "slug": "chronoseek-runtime",
            },
            "artifact": {
                "revision_sha": "abc123",
                "digest": "sha256:deadbeef",
            },
        }
    )

    assert metadata.endpoint == "https://runtime.example.com"
    assert metadata.chute_id == "chute-123"
    assert metadata.chute_slug == "chronoseek-runtime"
    assert metadata.artifact_id == "chronoseek/runtime"
    assert metadata.artifact_revision == "abc123"
    assert metadata.artifact_digest == "sha256:deadbeef"


def test_metadata_from_payload_shape():
    metadata = metadata_from_chutes_response(
        {
            "slug": "chronoseek-runtime",
            "name": "chronoseek-runtime",
            "image": "chronoseek/runtime",
            "revision": "abc123",
        }
    )

    assert metadata.chute_slug == "chronoseek-runtime"
    assert metadata.artifact_id == "chronoseek/runtime"
    assert metadata.artifact_revision == "abc123"


def test_merge_metadata_prefers_explicit_args():
    metadata = merge_metadata(
        from_args=RuntimeMetadata(
            endpoint="https://explicit.example.com",
            artifact_revision="explicit-revision",
        ),
        from_api=RuntimeMetadata(
            endpoint="https://deployed.example.com",
            chute_slug="deployed-runtime",
            artifact_revision="deployed-revision",
        ),
    )

    assert metadata.endpoint == "https://explicit.example.com"
    assert metadata.chute_slug == "deployed-runtime"
    assert metadata.artifact_revision == "explicit-revision"


def test_chain_metadata_accepts_chute_id_only():
    submission = MinerSubmission(hotkey="hk-1", chute_id="chute-123")

    assert submission.chute_id == "chute-123"


def test_endpoint_from_slug_uses_chutes_domain(monkeypatch):
    monkeypatch.setenv("CHUTES_BASE_DOMAIN", "chutes.ai")

    assert (
        resolve_submission_endpoint(
            MinerSubmission(hotkey="hk-1", chute_slug="chronoseek-runtime"),
            chutes_base_domain="chutes.ai",
        )
        == "https://chronoseek-runtime.chutes.ai"
    )


def test_submission_matches_expected_metadata():
    actual = MinerSubmission(
        hotkey="hk-1",
        endpoint="https://runtime.example.com/",
        chute_id="chute-123",
        chute_slug="chronoseek-runtime",
        artifact_revision="abc123",
    )

    assert actual.hotkey == "hk-1"
    assert str(actual.endpoint).rstrip("/") == "https://runtime.example.com"
    assert actual.chute_id == "chute-123"
    assert actual.chute_slug == "chronoseek-runtime"
    assert actual.artifact_revision == "abc123"


def test_chute_id_only_metadata_is_not_routable_yet():
    submission = MinerSubmission(hotkey="hk-1", chute_id="chute-123")

    assert (
        resolve_submission_endpoint(submission, chutes_base_domain="chutes.ai") is None
    )


def test_deploy_wrapper_explicit_metadata_normalizes_endpoint():
    metadata = explicit_metadata(DummyConfig())

    assert metadata.endpoint == "https://explicit.example.com"
    assert metadata.chute_id == "chute-123"
    assert metadata.chute_slug == "chronoseek-runtime"
    assert metadata.artifact_revision == "abc123"


def test_deploy_wrapper_miner_command_contains_commit_metadata():
    command = miner_command(
        RuntimeMetadata(
            endpoint="https://runtime.example.com",
            chute_id="chute-123",
            chute_slug="chronoseek-runtime",
            artifact_revision="abc123",
        ),
        DummyConfig(),
    )

    assert command[:4] == ["poetry", "run", "python", "miner.py"]
    assert "--wallet.name" not in command
    assert "--wallet.hotkey" not in command
    assert "--wallet.path" not in command
    assert "--subtensor.network" not in command
    assert "--netuid" not in command
    assert "--endpoint" in command
    assert "https://runtime.example.com" in command
    assert "--chute-id" in command
    assert "chute-123" in command
    assert "--chute-slug" in command
    assert "chronoseek-runtime" in command
    assert command.count("--capability") == 2


def test_metadata_with_chute_slug_prefers_generated_slug():
    metadata = metadata_with_chute_slug(
        RuntimeMetadata(
            endpoint="https://runtime.example.com",
            chute_id="chute-123",
            chute_slug="chronoseek-chronoseek-runtime",
            artifact_id="image-123",
            artifact_revision="rev-1",
        ),
        "chronoseek-chronoseek-runtime-20260510143015999",
    )

    assert metadata.endpoint == "https://runtime.example.com"
    assert metadata.chute_id == "chute-123"
    assert metadata.chute_slug == "chronoseek-chronoseek-runtime-20260510143015999"
    assert metadata.artifact_id == "image-123"
    assert metadata.artifact_revision == "rev-1"


def test_chute_definition_metadata_uses_chutes_public_slug():
    class DummyImage:
        uid = "image-123"
        tag = "rev-1"

    class DummyChute:
        uid = "chute-123"
        _username = "chronoseek"
        name = "chronoseek-runtime"
        _chronoseek_chute_slug = "chronoseek-chronoseek-runtime-20260510143015999"
        revision = "rev-2"
        image = DummyImage()

    metadata = metadata_from_chute_object(DummyChute())

    assert metadata.chute_id == "chute-123"
    assert metadata.chute_slug == "chronoseek-chronoseek-runtime-20260510143015999"
    assert metadata.artifact_id == "image-123"
    assert metadata.artifact_revision == "rev-2"


def test_runtime_name_uses_timestamp():
    assert resolve_chute_api_name("chronoseek-runtime") == "ChronoSeek-runtime"
    assert (
        resolve_chute_api_runtime_name(
            "chronoseek-runtime",
            "20260510143015999",
        )
        == "ChronoSeek-runtime-20260510143015999"
    )
    assert (
        resolve_chute_slug("chronoseek", "chronoseek-runtime", "20260510143015999")
        == "chronoseek-chronoseek-runtime-20260510143015999"
    )
    assert resolve_chute_display_name("chronoseek-runtime") == "ChronoSeek Runtime"


def test_resolve_chute_logo_url_uses_chronoseek_default():
    class ChuteWithoutLogo:
        pass

    assert (
        resolve_chute_logo_url(ChuteWithoutLogo()) == "https://chronoseek.org/logo.png"
    )


def test_resolve_chute_logo_url_uses_chute_override():
    class ChuteWithLogo:
        _chronoseek_logo_url = "https://example.com/logo.png"

    assert resolve_chute_logo_url(ChuteWithLogo()) == "https://example.com/logo.png"


def test_upload_logo_via_api_downloads_and_uploads_logo(monkeypatch):
    calls = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            calls.append(("get", url, headers))
            return chutes_deployment.httpx.Response(
                200,
                content=b"png-bytes",
                headers={"Content-Type": "image/png"},
                request=chutes_deployment.httpx.Request("GET", url),
            )

        async def post(self, url, files, headers):
            calls.append(("post", url, files, headers))
            return chutes_deployment.httpx.Response(
                200,
                json={"logo_id": "logo-123"},
                request=chutes_deployment.httpx.Request("POST", url),
            )

    monkeypatch.setattr(chutes_deployment.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        chutes_deployment,
        "chutes_auth_headers_from_env",
        lambda **kwargs: {"Authorization": "cpk_test"},
    )

    logo_id = asyncio.run(
        upload_logo_via_api(
            api_base_url="https://api.chutes.ai",
            logo_url="https://chronoseek.org/logo.png",
        )
    )

    assert logo_id == "logo-123"
    assert calls[1] == (
        "get",
        "https://chronoseek.org/logo.png",
        {"Accept": "image/*"},
    )
    assert calls[2][0] == "post"
    assert calls[2][1] == "https://api.chutes.ai/logos/"
    assert calls[2][2]["logo"][0] == "logo.png"
    assert calls[2][2]["logo"][1] == b"png-bytes"
    assert calls[2][2]["logo"][2] == "image/png"
    assert calls[2][3] == {"Authorization": "cpk_test"}


def test_apply_runtime_name_updates_chute_and_image_ids():
    class DummyImage:
        username = "chronoseek"
        name = "chronoseek-runtime-dev"
        tag = "abcdef1234567"
        _uid = "old-image"

        @property
        def uid(self):
            return self._uid

    class DummyChute:
        _uid = "old-chute"
        _username = "chronoseek"
        _name = "chronoseek-runtime"
        _readme = "Base readme."
        revision = "abcdef1234567"
        image = DummyImage()

        @property
        def uid(self):
            return self._uid

        @property
        def name(self):
            return self._name

        @property
        def readme(self):
            return self._readme

    chute = apply_runtime_name(
        DummyChute(),
        "chronoseek-runtime",
        chute_slug="chronoseek-chronoseek-runtime-20260510143015999",
        chute_display_name="ChronoSeek Runtime",
    )

    assert chute.name == "ChronoSeek-runtime"
    assert (
        chute._chronoseek_chute_slug
        == "chronoseek-chronoseek-runtime-20260510143015999"
    )
    assert chute.image.name == "chronoseek-chronoseek-runtime-20260510143015999-1234567"
    assert "Display label: `ChronoSeek Runtime`" in chute.readme
    assert (
        "Runtime slug: `chronoseek-chronoseek-runtime-20260510143015999`"
        in chute.readme
    )
    assert "Hotkey" not in chute.readme
    assert chute.uid != "old-chute"
    assert chute.image.uid != "old-image"


def test_chute_module_ref_rejects_file_paths():
    try:
        require_chute_module_ref("path/to/chute.py:chute")
    except ValueError as exc:
        assert "module ref" in str(exc)
    else:
        raise AssertionError("file-path chute refs should be rejected")


def test_image_build_payload_uses_chutes_image_object():
    from chutes.image import Image

    image = (
        Image(
            username="chronoseek",
            name="runtime",
            tag="rev-1",
            readme="Runtime readme.",
        )
        .from_base("parachutes/python:3.12")
        .run_command("echo hello")
    )

    data, files = image_build_form_payload(image, public=True, logo_id="logo-123")

    assert data["username"] == "chronoseek"
    assert data["name"] == "runtime"
    assert data["tag"] == "rev-1"
    assert data["public"] == "True"
    assert data["logo_id"] == "logo-123"
    assert "echo hello" in data["dockerfile"]
    assert data["image"]
    assert files["build_context"][0] == "chute.zip"
    assert files["build_context"][2] == "application/zip"


def test_chute_deploy_payload_uses_chutes_chute_object(tmp_path, monkeypatch):
    chute_file = tmp_path / "sample_chute.py"
    chute_file.write_text(
        """
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str

image = Image(username="chronoseek", name="runtime", tag="rev-1")
chute = Chute(
    username="chronoseek",
    name="runtime",
    tagline="Runtime",
    readme="Readme",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
    concurrency=1,
    revision="rev-1",
    allow_external_egress=True,
)

@chute.cord(public_api_path="/search", method="POST", input_schema=SearchRequest)
async def search(self, payload: SearchRequest):
    return {"ok": True}
""",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    payload = chute_deploy_payload(
        chute_ref="sample_chute:chute",
        public=True,
        logo_id="logo-123",
    )

    assert payload["name"] == "runtime"
    assert payload["public"] is True
    assert payload["logo_id"] == "logo-123"
    assert payload["filename"] == "sample_chute.py"
    assert payload["ref_str"] == "sample_chute:chute"
    assert payload["image"]
    assert payload["node_selector"]["gpu_count"] == 1
    assert payload["allow_external_egress"] is True
    assert payload["cords"][0]["public_api_path"] == "/search"


def test_chute_deploy_payload_includes_display_name_slug_and_runtime_readme(
    tmp_path,
    monkeypatch,
):
    chute_file = tmp_path / "sample_chute.py"
    chute_file.write_text(
        """
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

image = Image(username="chronoseek", name="runtime", tag="rev-1")
chute = Chute(
    username="chronoseek",
    name="chronoseek-runtime",
    tagline="Runtime",
    readme="Base readme.",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
    revision="rev-1",
)
""",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    payload = chute_deploy_payload(
        chute_ref="sample_chute:chute",
        public=True,
        chute_name="chronoseek-runtime",
        chute_slug="chronoseek-chronoseek-runtime-20260510143015999",
        chute_display_name="ChronoSeek Runtime",
        logo_id="logo-456",
    )

    assert payload["name"] == "ChronoSeek-runtime"
    assert payload["tagline"] == "Runtime"
    assert payload["logo_id"] == "logo-456"
    assert payload["slug"] == "chronoseek-chronoseek-runtime-20260510143015999"
    assert "Display label: `ChronoSeek Runtime`" in payload["readme"]
    assert (
        "Runtime slug: `chronoseek-chronoseek-runtime-20260510143015999`"
        in payload["readme"]
    )
    assert "Hotkey" not in payload["readme"]
    assert payload["image"]


def test_build_image_aborts_when_existing_image_overwrite_is_declined(monkeypatch):
    calls = []

    async def fake_get_image_via_api(**kwargs):
        calls.append(("get", kwargs))
        return {"image_id": "image-123"}

    async def fake_delete_image_via_api(**kwargs):
        calls.append(("delete", kwargs))

    async def fake_submit_image_build_via_api(**kwargs):
        calls.append(("submit", kwargs))
        return {"image_id": "image-123"}

    monkeypatch.setattr(
        chutes_deployment,
        "load_chute_object",
        lambda *args, **kwargs: DummyChutesChute(),
    )
    monkeypatch.setattr(chutes_deployment, "get_image_via_api", fake_get_image_via_api)
    monkeypatch.setattr(
        chutes_deployment,
        "delete_image_via_api",
        fake_delete_image_via_api,
    )
    monkeypatch.setattr(
        chutes_deployment,
        "submit_image_build_via_api",
        fake_submit_image_build_via_api,
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "n")

    try:
        asyncio.run(
            build_image_via_api(
                api_base_url="https://api.chutes.ai",
                chute_ref="chronoseek_chute:chute",
            )
        )
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("existing image should abort when overwrite is declined")

    assert [call[0] for call in calls] == ["get"]


def test_build_image_deletes_existing_image_after_yes_confirmation(monkeypatch):
    calls = []

    async def fake_get_image_via_api(**kwargs):
        calls.append(("get", kwargs))
        return {"image_id": "image-123"}

    async def fake_delete_image_via_api(**kwargs):
        calls.append(("delete", kwargs))
        return {"image_id": "image-123"}

    async def fake_submit_image_build_via_api(**kwargs):
        calls.append(("submit", kwargs))
        return {"image_id": "image-123"}

    monkeypatch.setattr(
        chutes_deployment,
        "load_chute_object",
        lambda *args, **kwargs: DummyChutesChute(),
    )
    monkeypatch.setattr(chutes_deployment, "get_image_via_api", fake_get_image_via_api)
    monkeypatch.setattr(
        chutes_deployment,
        "delete_image_via_api",
        fake_delete_image_via_api,
    )
    monkeypatch.setattr(
        chutes_deployment,
        "submit_image_build_via_api",
        fake_submit_image_build_via_api,
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "y")

    result = asyncio.run(
        build_image_via_api(
            api_base_url="https://api.chutes.ai",
            chute_ref="chronoseek_chute:chute",
            timeout_seconds=123.0,
        )
    )

    assert result == {"image_id": "image-123"}
    assert [call[0] for call in calls] == ["get", "delete", "submit"]
    assert calls[0][1]["image_id"] == "image-123"
    assert calls[1][1]["image_id"] == "image-123"
    assert calls[2][1]["timeout_seconds"] == 123.0


def test_build_image_deletes_existing_image_by_default_confirmation(monkeypatch):
    calls = []

    async def fake_get_image_via_api(**kwargs):
        calls.append(("get", kwargs))
        return {"image_id": "image-123"}

    async def fake_delete_image_via_api(**kwargs):
        calls.append(("delete", kwargs))
        return {"image_id": "image-123"}

    async def fake_submit_image_build_via_api(**kwargs):
        calls.append(("submit", kwargs))
        return {"image_id": "image-123"}

    monkeypatch.setattr(
        chutes_deployment,
        "load_chute_object",
        lambda *args, **kwargs: DummyChutesChute(),
    )
    monkeypatch.setattr(chutes_deployment, "get_image_via_api", fake_get_image_via_api)
    monkeypatch.setattr(
        chutes_deployment,
        "delete_image_via_api",
        fake_delete_image_via_api,
    )
    monkeypatch.setattr(
        chutes_deployment,
        "submit_image_build_via_api",
        fake_submit_image_build_via_api,
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    result = asyncio.run(
        build_image_via_api(
            api_base_url="https://api.chutes.ai",
            chute_ref="chronoseek_chute:chute",
        )
    )

    assert result == {"image_id": "image-123"}
    assert [call[0] for call in calls] == ["get", "delete", "submit"]


def test_deploy_wrapper_build_deploy_flow_invokes_chutes_api_boundaries(
    monkeypatch,
    capsys,
):
    calls = []

    async def fake_build_image_via_api(**kwargs):
        calls.append(("build", kwargs))
        return {"image_id": "image-123"}

    async def fake_upload_logo_via_api(**kwargs):
        calls.append(("logo", kwargs))
        return "logo-123"

    async def fake_stream_image_build_logs_via_api(**kwargs):
        calls.append(("logs", kwargs))

    async def fake_deploy_chute_via_api(**kwargs):
        calls.append(("deploy", kwargs))
        return {
            "chute_id": "chute-123",
            "slug": "chronoseek-chronoseek-runtime",
            "image": "image-123",
            "revision": "rev-1",
        }

    def fake_metadata_from_chute_definition(*args, **kwargs):
        return RuntimeMetadata(
            chute_id="chute-123",
            chute_slug=kwargs.get("chute_slug") or "chronoseek-runtime",
            artifact_id="image-123",
            artifact_revision="rev-1",
        )

    monkeypatch.setattr(
        deploy_chutes_runtime,
        "metadata_from_chute_definition",
        fake_metadata_from_chute_definition,
    )
    monkeypatch.setattr(
        deploy_chutes_runtime,
        "build_image_via_api",
        fake_build_image_via_api,
    )
    monkeypatch.setattr(
        deploy_chutes_runtime,
        "upload_logo_via_api",
        fake_upload_logo_via_api,
    )
    monkeypatch.setattr(
        deploy_chutes_runtime,
        "stream_image_build_logs_via_api",
        fake_stream_image_build_logs_via_api,
    )
    monkeypatch.setattr(
        deploy_chutes_runtime,
        "deploy_chute_via_api",
        fake_deploy_chute_via_api,
    )

    class DummyLoadedChute:
        _username = "chronoseek"
        name = "chronoseek-runtime"
        _chronoseek_logo_url = "https://chronoseek.org/logo.png"

    monkeypatch.setattr(
        deploy_chutes_runtime,
        "load_chute_object",
        lambda *args, **kwargs: DummyLoadedChute(),
    )

    monkeypatch.setattr(
        deploy_chutes_runtime,
        "resolve_runtime_timestamp",
        lambda: "20260510143015999",
    )
    monkeypatch.setattr(
        deploy_chutes_runtime.sys,
        "argv",
        [
            "deploy_chutes_runtime.py",
            "--build",
            "--deploy",
            "--chute-ref",
            "chronoseek_chute:chute",
            "--include-cwd",
            "--accept-fee",
            "--public",
        ],
    )

    assert asyncio.run(deploy_chutes_runtime.main_async()) == 0

    assert calls == [
        (
            "logo",
            {
                "api_base_url": "https://api.chutes.ai",
                "logo_url": "https://chronoseek.org/logo.png",
                "timeout_seconds": 120.0,
            },
        ),
        (
            "build",
            {
                "api_base_url": "https://api.chutes.ai",
                "chute_ref": "chronoseek_chute:chute",
                "include_cwd": True,
                "public": True,
                "overwrite_existing": None,
                "timeout_seconds": 900.0,
                "chute_name": "ChronoSeek-runtime-20260510143015999",
                "chute_slug": "chronoseek-chronoseek-runtime-20260510143015999",
                "chute_display_name": "ChronoSeek Runtime",
                "logo_id": "logo-123",
            },
        ),
        (
            "logs",
            {
                "api_base_url": "https://api.chutes.ai",
                "image_id": "image-123",
                "timeout_seconds": 900.0,
            },
        ),
        (
            "deploy",
            {
                "api_base_url": "https://api.chutes.ai",
                "chute_ref": "chronoseek_chute:chute",
                "public": True,
                "accept_fee": True,
                "timeout_seconds": 900.0,
                "chute_name": "ChronoSeek-runtime-20260510143015999",
                "chute_slug": "chronoseek-chronoseek-runtime-20260510143015999",
                "chute_display_name": "ChronoSeek Runtime",
                "logo_id": "logo-123",
            },
        ),
    ]
    output = capsys.readouterr().out
    assert '"chute_id": "chute-123"' in output
    assert '"chute_slug": "chronoseek-chronoseek-runtime-20260510143015999"' in output
    assert '"chute_slug": "chronoseek-chronoseek-runtime"' not in output
    assert "--chute-slug chronoseek-chronoseek-runtime-20260510143015999" in output
    assert "--wallet.name" not in output
    assert "--wallet.hotkey" not in output
