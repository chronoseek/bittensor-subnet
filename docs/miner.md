# Miner Guide

This guide is for operators running a ChronoSeek miner.

In `v2.0`, a miner does not serve a local HTTP endpoint from `miner.py`. A miner deploys a retrieval runtime to Chutes, then commits the runtime metadata on-chain. Validators read one permanent submission for each registered hotkey and query the submitted Chutes runtime for synthetic scoring.

## Prerequisites

- Python 3.12+
- Poetry
- A registered miner hotkey on the configured subnet
- A Chutes account and API key
- A Chutes runtime definition, usually copied from `chronoseek_chute.example.py`
- Optional Hugging Face token for model downloads and fewer rate limits
- Optional yt-dlp cookies for restricted YouTube/Vimeo videos

Install dependencies from the repository root:

```bash
poetry install
cp .env.example .env
```

## Required Environment Variables

For on-chain miner metadata commits:

| Variable | Required | Notes |
| --- | --- | --- |
| `WALLET_NAME` | Yes | Coldkey wallet name. |
| `HOTKEY_NAME` | Yes | Registered miner hotkey name. |
| `NETWORK` | Yes | Usually `finney`, `test`, or `local`. |
| `NETUID` | Yes | ChronoSeek subnet netuid. |
| `WALLET_PATH` | If non-default | Defaults to `~/.bittensor/wallets`. |

For Chutes runtime build/deploy:

| Variable | Required | Notes |
| --- | --- | --- |
| `CHUTES_API_KEY` | Yes | Used by `scripts/deploy_chutes_runtime.py`; the helper resolves the Chutes username from this key. |
| `HF_TOKEN` | Recommended | Used by CLIP/Whisper model downloads when present. |
| `YTDLP_COOKIES` | Optional | Cookies file copied into Chutes image when it points to a readable local file. |
| `YTDLP_COOKIES_BROWSER` | Optional | Browser cookie source used by yt-dlp where that browser profile is available. Chutes deployments should prefer `YTDLP_COOKIES`. |
| `YTDLP_DENO_PATH` | Optional | Explicit Deno path for yt-dlp challenge solving. The Chutes image sets this internally. |
| `YTDLP_NODE_PATH` | Optional | Explicit Node.js path for yt-dlp challenge solving. |

Minimal `.env` shape:

```env
NETWORK=finney
NETUID=<netuid>
WALLET_NAME=<miner-coldkey>
HOTKEY_NAME=<miner-hotkey>
WALLET_PATH=~/.bittensor/wallets

CHUTES_API_KEY=<chutes-api-key>
HF_TOKEN=<hugging-face-token>
YTDLP_COOKIES=/absolute/path/to/cookies.txt
```

## Runtime Setup

Create your local Chutes definition:

```bash
cp chronoseek_chute.example.py chronoseek_chute.py
```

Edit `chronoseek_chute.py` before deployment. This file is ignored by git and is where miner-specific package and GPU settings belong. The deploy helper resolves the Chutes username from `CHUTES_API_KEY` with `GET /users/me`.

Important fields:

| Field | Notes |
| --- | --- |
| `DEFAULT_CHUTE_BASE_NAME` | Constant in `chronoseek/constants.py`; `CHUTE_NAME` is generated from it. |
| `CHRONOSEEK_PACKAGE` | Package or git URL installed into the Chutes image. Use your fork if you changed runtime code. |
| `RUNTIME_REVISION` | Derived from the current git commit when available. |
| `node_selector` | GPU requirements. Start conservatively. |
| `concurrency` | Per-instance request concurrency. The template uses `5` to keep validator fanout responsive. |
| `max_instances` | Maximum autoscaled Chutes instances. The template uses `3`. |
| `scaling_threshold` | Utilization threshold for adding capacity. The template uses `0.5`. |
| `shutdown_after_seconds` | Idle time before Chutes shuts an instance down. The template uses `300`. |
| `allow_external_egress=True` | Required so the runtime can fetch validator task videos. |

The example runtime exposes:

- `/health`
- `/search`

Both are native Chutes cords. The runtime does not start a separate FastAPI server inside Chutes.

## Local-First Checklist

Run these checks before calling production Chutes APIs:

```bash
poetry run pytest tests/unit
poetry run python scripts/verify_miner_retrieval.py
poetry run python scripts/test_chutes_runtime_local.py --print-commands
```

For local container testing, `scripts/test_chutes_runtime_local.py` defaults to `chronoseek_chute_local:chute`. That local template installs the current working-tree source into the test image, so it is useful for checking uncommitted runtime changes. The production deployment helper defaults to `chronoseek_chute:chute`, which should point at the immutable package or git revision you intend to commit on-chain.

## Deploy Runtime

Build and deploy through the Chutes API:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee \
  --artifact-id chronoseek-runtime
```

The helper prints normalized metadata and a suggested `miner.py` command. With resolved Chutes username `alice`, the submitted slug looks like:

```text
alice-chronoseek-runtime-<timestamp>
```

Use the exact slug printed by the deployment helper.

You can print the metadata and suggested commit command without deploying by omitting `--build` and `--deploy`:

```bash
poetry run python scripts/deploy_chutes_runtime.py \
  --chute-ref chronoseek_chute:chute
```

If the image already exists, the deploy helper prompts before deleting and rebuilding it. For non-interactive behavior:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build \
  --chute-ref chronoseek_chute:chute \
  --overwrite-existing-image
```

Rare advanced redeploys that reuse a prebuilt Chutes image should edit
`chronoseek_chute.py` directly for that one-off flow. Do this only when the
container image contents are unchanged and the prebuilt image already has
Chutes status `built and pushed`.

## Commit Runtime Metadata

After deployment, commit the runtime submission on-chain:

```bash
poetry run python miner.py \
  --wallet.name <miner-coldkey> \
  --wallet.hotkey <miner-hotkey> \
  --chute-slug alice-chronoseek-runtime-<timestamp> \
  --artifact-revision <immutable-revision>
```

Miner runtime submissions are permanent per hotkey. A hotkey can submit only once; to submit a different runtime later, register and use a new miner hotkey. Validators disqualify hotkeys with multiple revealed submissions and assign them zero score.

For tests that require replacing a submitted runtime, set `ENFORCE_ONE_HOTKEY_ONE_SUBMISSION=0` for both the miner and validator. The miner then skips its duplicate check, and the validator uses the newest revealed submission without applying the duplicate-submission penalty. Production behavior remains enabled by default.

Before submitting, you can check whether the selected hotkey already has revealed commit data:

```bash
poetry run python miner.py \
  --wallet.name <miner-coldkey> \
  --wallet.hotkey <miner-hotkey> \
  --network <finney-or-test> \
  --netuid <netuid> \
  --check-existing
```

This command prints the hotkey, netuid, registration state, and any revealed commitments as JSON. It is read-only and exits without submitting new metadata.

You can also include Chutes and artifact metadata:

```bash
poetry run python miner.py \
  --wallet.name <miner-coldkey> \
  --wallet.hotkey <miner-hotkey> \
  --chute-id <chute-deployment-id> \
  --chute-slug alice-chronoseek-runtime-<timestamp> \
  --artifact-id chronoseek-runtime \
  --artifact-revision <immutable-revision> \
  --artifact-digest <digest>
```

Current validators require either `--endpoint` or `--chute-slug` to route requests. `--chute-id` is useful metadata, but it is not routable by itself yet.

## Axon Serving Track (Fallback)

An alternative to Chutes deployment: serve the miner runtime directly over your own Bittensor axon instead of deploying an image to Chutes.

Validators pick a base URL per hotkey per refresh cycle: a committed Chutes endpoint always wins when one resolves; the axon address is only used when there is no Chutes commitment resolvable at all (missing commitment, or a `--chute-id`-only commitment with no routable `--endpoint`/`--chute-slug`). Whichever path is selected, a failed/timed-out request scores 0 for that round with no fallback to the other path.

```bash
poetry run python miner_axon.py \
  --wallet.name <miner-coldkey> \
  --wallet.hotkey <miner-hotkey> \
  --network <finney-or-test> \
  --netuid <netuid> \
  --axon-external-ip <your-public-ip> \
  --axon-port 8091
```

This serves `chronoseek.miner.runtime:app` (the same `/health`/`/search` contract, Epistula-signed, used by the Chutes path) directly with uvicorn, and separately publishes `<external-ip>:<port>` on-chain via the `ServeAxon` intent so validators can resolve it. Bittensor 11 removed the old `bt.axon` networking stack, so `--axon-external-ip` must be supplied explicitly — there is no external-IP auto-detection.

This process does not commit any metadata on-chain; that only happens via `miner.py`. A validator only tries this hotkey's axon when it finds no Chutes commitment for it at all.

## Video Download Behavior

Miner Chutes must handle synthetic requests from validators. Owner-run Chutes runtimes are forked from miner runtimes and may also receive non-Hippius videos.

The runtime uses the shared downloader:

1. Hippius URLs use the Hippius S3-compatible API.
2. YouTube, Vimeo, and other extractor-platform URLs use `yt-dlp`.
3. Direct media URLs use raw HTTP.
4. If the first strategy fails, the runtime falls back to another supported strategy.
5. If every strategy fails, `/search` returns a structured `VIDEO_FETCH_FAILED` error.

For restricted videos in Chutes deployments, prefer a cookies file:

```env
YTDLP_COOKIES=/absolute/path/to/cookies.txt
```

For local runs where a browser profile is available, `yt-dlp` can also use:

```env
YTDLP_COOKIES_BROWSER=chrome:Default
```

During Chutes image build, a readable `YTDLP_COOKIES` file is copied into the image and the environment is rewritten to the container path.

If validator task clips come from a non-default Hippius-compatible host, update
the Hippius defaults in `chronoseek/constants.py`.

## Common Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `MIN_VALIDATOR_STAKE` | `10000` | Minimum validator stake enforced by runtime auth paths that use hotkey headers. |
| `DEFAULT_CHUTES_HF_HOME` | `/data/huggingface` | Hugging Face cache path inside Chutes images. |
| `YTDLP_DENO_PATH` | Empty locally; `/opt/deno/bin/deno` in Chutes images | Deno path for yt-dlp challenge solving. |
| `YTDLP_NODE_PATH` | Empty | Optional Node.js path for yt-dlp challenge solving. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

## Local Checks

Use the local Chutes flow before production build/deploy. See [Owner and development guide](./development.md) for local runtime checks and live module tests.
