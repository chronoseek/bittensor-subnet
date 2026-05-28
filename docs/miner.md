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
| `CHUTES_API_KEY` | Yes | Used by `scripts/deploy_chutes_runtime.py`. |
| `HF_TOKEN` | Recommended | Used by CLIP/Whisper model downloads when present. |
| `YTDLP_COOKIES` | Optional | Cookies file copied into Chutes image when it points to a readable local file. |
| `YTDLP_COOKIES_BROWSER` | Optional | Browser cookie source used by yt-dlp where that browser profile is available. Chutes deployments should prefer `YTDLP_COOKIES`. |

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

Edit `chronoseek_chute.py` before deployment. The examples assume Chutes account `alice`.

Important fields:

| Field | Notes |
| --- | --- |
| `CHUTES_ACCOUNT` | Chutes account namespace, for example `alice`. This is not your Bittensor identity. |
| `CHUTE_BASE_NAME` | Defaults to `chronoseek-runtime`. |
| `CHRONOSEEK_PACKAGE` | Package or git URL installed into the Chutes image. Use your fork if you changed runtime code. |
| `RUNTIME_REVISION` | Git SHA, image version, or immutable runtime label. |
| `node_selector` | GPU requirements. Start conservatively. |
| `concurrency` | Keep low initially because video retrieval is resource-heavy. |
| `allow_external_egress=True` | Required so the runtime can fetch validator task videos. |

The example runtime exposes:

- `/health`
- `/search`

Both are native Chutes cords. The runtime does not start a separate FastAPI server inside Chutes.

## Deploy Runtime

Build and deploy through the Chutes API:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee \
  --artifact-id chronoseek-runtime
```

The helper prints normalized metadata and a suggested `miner.py` command. With Chutes account `alice`, the submitted slug looks like:

```text
alice-chronoseek-runtime-<timestamp>
```

Use the exact slug printed by the deployment helper.

If the image already exists, the deploy helper prompts before deleting and rebuilding it. For non-interactive behavior:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build \
  --chute-ref chronoseek_chute:chute \
  --overwrite-existing-image
```

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

## Common Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `CHUTES_BASE_DOMAIN` | `chutes.ai` | Used by validators to resolve submitted Chutes slugs. |
| `MIN_VALIDATOR_STAKE` | `10000` | Minimum validator stake enforced by runtime auth paths that use hotkey headers. |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache path. |
| `YTDLP_DENO_PATH` | `/opt/deno/bin/deno` | Deno path for yt-dlp challenge solving in Chutes images. |
| `YTDLP_NODE_PATH` | Empty | Optional Node.js path for yt-dlp challenge solving. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

## Local Checks

Use the local Chutes flow before production build/deploy. See [Owner and development guide](./development.md) for local runtime checks and live module tests.
