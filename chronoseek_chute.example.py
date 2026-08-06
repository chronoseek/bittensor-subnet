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

import asyncio
import os
import shutil
import subprocess
import tempfile
import threading

from dotenv import load_dotenv
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

from chronoseek.constants import (
    DEFAULT_CHRONOSEEK_LOGO_URL,
    DEFAULT_CHUTE_BASE_NAME,
    DEFAULT_CHUTES_HF_HOME,
    DEFAULT_CHUTES_YTDLP_DENO_PATH,
)
from chronoseek.protocol_models import (
    ProofOfAccessRequest,
    ProofOfAccessResponse,
    VideoSearchRequest,
    VideoSearchResponse,
)


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def resolve_runtime_revision() -> str:
    """Resolve runtime revision from the current git commit SHA."""

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


def prepare_runtime_ytdlp_cookies() -> None:
    """Copy baked yt-dlp cookies to a private, writable runtime file."""

    source_path = os.path.expanduser(os.getenv("YTDLP_COOKIES", "").strip())
    if not source_path or not os.path.isfile(source_path):
        return

    runtime_directory = tempfile.mkdtemp(prefix="chronoseek-ytdlp-cookies-")
    runtime_path = os.path.join(runtime_directory, "cookies.txt")
    shutil.copyfile(source_path, runtime_path)
    os.chmod(runtime_path, 0o600)
    os.environ["YTDLP_COOKIES"] = runtime_path


# Placeholder required by Chutes SDK object construction. The deploy helper
# replaces it with the username resolved from CHUTES_API_KEY before build/deploy.
CHUTES_USERNAME = "<change-me>"
CHUTE_BASE_NAME = DEFAULT_CHUTE_BASE_NAME
CHUTE_NAME = CHUTE_BASE_NAME
CHRONOSEEK_LOGO_URL = DEFAULT_CHRONOSEEK_LOGO_URL
RUNTIME_REVISION = resolve_runtime_revision()
IMAGE_NAME = resolve_image_name(CHUTE_NAME, RUNTIME_REVISION)
# Chutes API enforces <=32 chars for image tags.
IMAGE_TAG = RUNTIME_REVISION[:32]
PREBUILT_IMAGE_ID = os.getenv("CHUTES_PREBUILT_IMAGE_ID", "").strip()

# The deployed Chutes image needs the ChronoSeek package and native video tools.
# Use a public git URL, a private URL with deploy credentials, or replace this
# with your own image/package install command. Do not commit embedded secrets.
CHRONOSEEK_PACKAGE = "git+https://github.com/chronoseek/bittensor-subnet.git"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
IMAGE_YTDLP_DENO_PATH = DEFAULT_CHUTES_YTDLP_DENO_PATH
# No default here: the deploy wrapper (chutes_ytdlp_cookie_file_context in
# chronoseek/chutes/deployment.py) bakes a real cookies.txt file into every
# build automatically, which downloader.py prefers over any browser fallback.
# A "chrome:Default" default would be actively wrong here anyway - miner GPU
# instances have no Chrome profile to read.
YTDLP_COOKIES_BROWSER = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()
YTDLP_DENO_PATH = IMAGE_YTDLP_DENO_PATH


image = (
    Image(
        username=CHUTES_USERNAME,
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
    .with_env("HF_HOME", DEFAULT_CHUTES_HF_HOME)
    .with_env("DENO_INSTALL", "/opt/deno")
    .with_env("PATH", "/opt/deno/bin:$PATH")
    .with_env("YTDLP_COOKIES_BROWSER", YTDLP_COOKIES_BROWSER)
    .with_env("YTDLP_DENO_PATH", YTDLP_DENO_PATH)
    .run_command(
        "apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && "
        "apt-get install -y --no-install-recommends "
        "ca-certificates curl ffmpeg git libgl1 libglib2.0-0 unzip && "
        "apt-get autoremove -y && "
        "apt-get clean && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .run_command(
        "rm -rf /opt/nvidia/nsight-compute /opt/nvidia/nsight-systems"
    )
    .run_command(
        "curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/opt/deno sh && "
        "chmod -R a+rx /opt/deno"
    )
    .run_command(
        "mkdir -p /tmp/pip-cache /tmp/uv-cache /tmp/.cache "
        f"{DEFAULT_CHUTES_HF_HOME} && "
        "chmod -R a+rwx /tmp/pip-cache /tmp/uv-cache /tmp/.cache /data"
    )
    .run_command("pip install --upgrade pip")
    .run_command("pip install 'torch==2.11.0' --torch-backend=cu128")
    .run_command(f"pip install '{CHRONOSEEK_PACKAGE}' --torch-backend=cu128")
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
    username=CHUTES_USERNAME,
    name=CHUTE_NAME,
    tagline="ChronoSeek",
    readme=(
        "# ChronoSeek Runtime\n\n"
        "ChronoSeek runtime for validator evaluation. Exposes `/health` and `/search`."
    ),
    image=PREBUILT_IMAGE_ID or image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        include=["pro_6000"],
        # Optional cost/scheduling controls:
        # max_hourly_price_per_gpu=1.50,
        # exclude=["k80"],
    ),
    concurrency=5,
    max_instances=3,
    scaling_threshold=0.5,
    # Raised from the SDK default (300s) to reduce how often a scheduled
    # instance shuts down from idling between validator health checks. This
    # only helps keep an already-scheduled instance alive longer; it has no
    # effect on how often the Chutes marketplace grants an instance in the
    # first place, and idle GPU time is billed against the deploying account.
    shutdown_after_seconds=86400,
    revision=RUNTIME_REVISION,
    # The runtime must fetch arbitrary validator task videos.
    allow_external_egress=True,
    # Chutes production deployments require TEE-enabled chutes.
    tee=True,
)
chute._chronoseek_logo_url = CHRONOSEEK_LOGO_URL


@chute.on_startup()
async def initialize_chronoseek(self):
    """Initialize ChronoSeek once per Chutes instance."""

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = os.path.expanduser(hf_token.strip())
    os.environ["HF_HOME"] = os.path.expanduser(DEFAULT_CHUTES_HF_HOME)
    prepare_runtime_ytdlp_cookies()

    deno_path = os.path.expanduser(os.getenv("YTDLP_DENO_PATH", "").strip())
    if not deno_path or not os.path.exists(deno_path):
        os.environ["YTDLP_DENO_PATH"] = IMAGE_YTDLP_DENO_PATH

    node_path = os.path.expanduser(os.getenv("YTDLP_NODE_PATH", "").strip())
    if node_path and not os.path.exists(node_path):
        os.environ.pop("YTDLP_NODE_PATH", None)

    from chronoseek.miner import runtime as chronoseek_runtime

    # Chutes activation only starts after startup hooks return. Model loading can
    # take long enough to make activation fail, so warm the runtime in the
    # background and let /health report readiness when initialization finishes.
    threading.Thread(
        target=chronoseek_runtime.initialize_runtime,
        name="chronoseek-runtime-initialize",
        daemon=True,
    ).start()


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
async def search(self, payload: VideoSearchRequest) -> dict:
    """Run ChronoSeek search as a native Chutes SDK cord.

    Chutes native cords do not pass arbitrary public HTTP headers to user code,
    so validator identity is enforced by Chutes API access for this deployment
    path.

    Runs off the cord's own event loop via a worker thread, serialized behind
    a semaphore - execute_search does a real video download plus GPU
    inference, and running it inline previously blocked /health (and
    Chutes' own verification probe) for the whole pipeline duration, causing
    the probe to time out and the instance to get torn down under load.
    """

    from chronoseek.miner import runtime as chronoseek_runtime

    async with chronoseek_runtime._inference_semaphore:
        return await asyncio.to_thread(
            chronoseek_runtime.execute_search,
            payload,
            caller_hotkey=None,
            enforce_validator_auth=False,
        )


@chute.cord(
    public_api_path="/proof-of-access",
    method="POST",
    input_schema=ProofOfAccessRequest,
    output_schema=ProofOfAccessResponse,
)
async def proof_of_access(self, payload: ProofOfAccessRequest) -> dict:
    """Run ChronoSeek proof-of-access as a native Chutes SDK cord.

    Chutes native cords do not pass arbitrary public HTTP headers to user code,
    so validator identity is enforced by Chutes API access for this deployment
    path.

    Runs off the cord's own event loop the same way search() does - see its
    docstring for why.
    """

    from chronoseek.miner import runtime as chronoseek_runtime

    async with chronoseek_runtime._inference_semaphore:
        return await asyncio.to_thread(
            chronoseek_runtime.execute_proof_of_access,
            payload,
            caller_hotkey=None,
            enforce_validator_auth=False,
        )
