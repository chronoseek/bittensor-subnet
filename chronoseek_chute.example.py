"""Example Chutes definition for a ChronoSeek miner runtime.

Copy this file to `chronoseek_chute.py`, edit the values marked CHANGE_ME, then
deploy from the repository root with:

    poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
      --chute-ref chronoseek_chute:chute \
      --accept-fee \
      --artifact-id chronoseek-runtime

The Chutes SDK loader expects module refs from the working directory, so the
local copy must live in the subnet root and is referenced as
`chronoseek_chute:chute`.
"""

import os
import subprocess

from dotenv import load_dotenv
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

from chronoseek.protocol_models import VideoSearchRequest, VideoSearchResponse


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def resolve_runtime_revision() -> str:
    """Resolve runtime revision from env or git commit SHA."""

    env_revision = os.getenv("RUNTIME_REVISION", "").strip()
    if env_revision:
        return env_revision

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        # Fallback for non-git packaging contexts.
        return "dev"


def resolve_image_name(base_name: str, runtime_revision: str) -> str:
    """Build image name as <base-name>-<last7sha>."""

    short_rev = str(runtime_revision or "").strip().lower()[-7:]
    if not short_rev:
        return base_name
    return f"{base_name}-{short_rev}"


# Required by Chutes SDK image/chute object construction. This is not part of
# ChronoSeek miner identity.
CHUTES_ACCOUNT = "CHANGE_ME"
CHUTE_BASE_NAME = "chronoseek-runtime"
CHUTE_NAME = CHUTE_BASE_NAME
CHRONOSEEK_LOGO_URL = "https://chronoseek.org/logo.png"
RUNTIME_REVISION = resolve_runtime_revision()
IMAGE_NAME = resolve_image_name(CHUTE_NAME, RUNTIME_REVISION)
# Chutes API enforces <=32 chars for image tags.
IMAGE_TAG = RUNTIME_REVISION[:32]

# The deployed Chutes image needs the ChronoSeek package and native video tools.
# Use a public git URL, a private URL with deploy credentials, or replace this
# with your own image/package install command. Do not commit embedded secrets.
CHRONOSEEK_PACKAGE = "git+https://github.com/chronoseek/bittensor-subnet.git"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
IMAGE_YTDLP_DENO_PATH = "/opt/deno/bin/deno"
YTDLP_COOKIES_BROWSER = (
    os.getenv("YTDLP_COOKIES_BROWSER", "chrome:Default").strip()
    or "chrome:Default"
)
YTDLP_DENO_PATH = IMAGE_YTDLP_DENO_PATH


image = (
    Image(
        username=CHUTES_ACCOUNT,
        name=IMAGE_NAME,
        tag=IMAGE_TAG,
        readme="ChronoSeek miner runtime.",
    )
    .from_base("parachutes/python:3.12")
    .set_user("root")
    # Chutes injects a final `pip install chutes==...` as the `chutes` user.
    # Keep pip/uv caches out of /home/chutes, which can be root-owned during
    # image finalization.
    .with_env("PIP_NO_CACHE_DIR", "1")
    .with_env("UV_NO_CACHE", "1")
    .with_env("PIP_CACHE_DIR", "/tmp/pip-cache")
    .with_env("UV_CACHE_DIR", "/tmp/uv-cache")
    .with_env("XDG_CACHE_HOME", "/tmp/.cache")
    .with_env("HF_HOME", "/data/huggingface")
    .with_env("DENO_INSTALL", "/opt/deno")
    .with_env("PATH", "/opt/deno/bin:$PATH")
    .with_env("YTDLP_COOKIES_BROWSER", YTDLP_COOKIES_BROWSER)
    .with_env("YTDLP_DENO_PATH", YTDLP_DENO_PATH)
    .run_command(
        "apt-get update && "
        "apt-get install -y --no-install-recommends "
        "ca-certificates curl ffmpeg git libgl1 libglib2.0-0 unzip && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .run_command(
        "curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/opt/deno sh && "
        "chmod -R a+rx /opt/deno"
    )
    .run_command(
        "mkdir -p /tmp/pip-cache /tmp/uv-cache /tmp/.cache /data/huggingface && "
        "chmod -R a+rwx /tmp/pip-cache /tmp/uv-cache /tmp/.cache /data"
    )
    .run_command("pip install --upgrade pip")
    .run_command(f"pip install '{CHRONOSEEK_PACKAGE}'")
    # Chutes later adds the `chutes` user to group root, then runs another
    # package install as that user. Make root-installed packages group-writable
    # so that finalization can update shared dependencies such as uvicorn.
    .run_command(
        "chmod -R g+rwX /usr/local/lib /usr/local/bin "
        "/usr/local/share /usr/local/share/man || true"
    )
)

if HF_TOKEN:
    image = image.with_env("HF_TOKEN", HF_TOKEN)


chute = Chute(
    username=CHUTES_ACCOUNT,
    name=CHUTE_NAME,
    tagline="ChronoSeek",
    readme=(
        "# ChronoSeek Runtime\n\n"
        "ChronoSeek runtime for validator evaluation. Exposes `/health` and `/search`."
    ),
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        # Optional cost/scheduling controls:
        # max_hourly_price_per_gpu=1.50,
        # include=["a100"],
        # exclude=["k80"],
    ),
    concurrency=1,
    revision=RUNTIME_REVISION,
    # The runtime must fetch arbitrary validator task videos.
    allow_external_egress=True,
)
chute._chronoseek_logo_url = CHRONOSEEK_LOGO_URL


@chute.on_startup()
async def initialize_chronoseek(self):
    """Initialize ChronoSeek once per Chutes instance."""

    import sys

    for env_name in ("HF_TOKEN", "HF_HOME"):
        env_value = os.getenv(env_name)
        if env_value:
            os.environ[env_name] = os.path.expanduser(env_value.strip())

    deno_path = os.path.expanduser(os.getenv("YTDLP_DENO_PATH", "").strip())
    if not deno_path or not os.path.exists(deno_path):
        os.environ["YTDLP_DENO_PATH"] = IMAGE_YTDLP_DENO_PATH

    node_path = os.path.expanduser(os.getenv("YTDLP_NODE_PATH", "").strip())
    if node_path and not os.path.exists(node_path):
        os.environ.pop("YTDLP_NODE_PATH", None)

    # Chutes finalization installs substrate-interface, which brings in
    # scalecodec. Bittensor's async substrate stack expects cyscale in that
    # namespace, so repair the shared package namespace before importing runtime.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "scalecodec", "cyscale"],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "cyscale>=0.3.3,<1.0.0",
        ],
        check=True,
    )

    from chronoseek.miner import runtime as chronoseek_runtime

    # Initialize model pipeline, Bittensor metagraph auth, and runtime globals.
    chronoseek_runtime.initialize_runtime()


@chute.cord(
    public_api_path="/health",
    public_api_method="GET",
    method="GET",
)
async def health(self):
    """Return Chutes runtime health without invoking video inference."""

    from chronoseek.miner import runtime as chronoseek_runtime

    return chronoseek_runtime.health_payload()


@chute.cord(
    public_api_path="/search",
    method="POST",
    input_schema=VideoSearchRequest,
    output_schema=VideoSearchResponse,
)
async def search(self, payload: VideoSearchRequest) -> VideoSearchResponse:
    """Run ChronoSeek search as a native Chutes SDK cord.

    Chutes native cords do not pass arbitrary public HTTP headers to user code,
    so validator identity is enforced by Chutes API access for this deployment
    path.
    """

    from chronoseek.miner import runtime as chronoseek_runtime

    return chronoseek_runtime.execute_search(
        payload,
        caller_hotkey=None,
        enforce_validator_auth=False,
    )
