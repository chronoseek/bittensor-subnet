# Validator Guide

This guide is for operators running a ChronoSeek validator.

Validators generate synthetic video-search tasks, query miner Chutes runtimes, score responses, and set on-chain weights. In hardened mode, validators also create private task clips, upload them to Hippius, and send miners only the clipped task URL plus query.

## Prerequisites

- Python 3.12+
- Poetry
- A registered validator hotkey on the configured subnet
- `ffmpeg` available in `PATH` when hardened task clipping/compression is enabled
- A Hugging Face token, unless `TASK_DATASET_PATH` points to a local ActivityNet-style dataset
- A Chutes API key for querying private miner runtimes
- Hippius S3 credentials when hardened task generation is enabled

Install dependencies from the repository root:

```bash
poetry install
cp .env.example .env
```

## Required Environment Variables

Base validator:

| Variable | Required | Notes |
| --- | --- | --- |
| `WALLET_NAME` | Yes | Coldkey wallet name. |
| `HOTKEY_NAME` | Yes | Registered validator hotkey name. |
| `NETWORK` | Yes | Usually `finney`, `test`, or `local`. |
| `NETUID` | Yes | ChronoSeek subnet netuid. |
| `WALLET_PATH` | If non-default | Defaults to `~/.bittensor/wallets`. |
| `CHUTES_API_KEY` | Yes | Used to query private Chutes runtimes. |
| `HF_TOKEN` | Yes, unless `TASK_DATASET_PATH` is set | Required for Hugging Face ActivityNet loading. |

Hardened task generation:

| Variable | Required when | Notes |
| --- | --- | --- |
| `ENABLE_HARDENED_TASKS=1` | Enabling hardened tasks | Generates private clipped task artifacts. |
| `VALIDATOR_TASK_SECRET` | `ENABLE_HARDENED_TASKS=1` | Set a long random secret. Used for private deterministic sampling. |
| `HIPPIUS_S3_ACCESS_KEY_ID` | `ENABLE_HARDENED_TASKS=1` | Hippius S3 access key ID. |
| `HIPPIUS_S3_SECRET_ACCESS_KEY` | `ENABLE_HARDENED_TASKS=1` | Hippius S3 secret key. |
| `HIPPIUS_S3_BUCKET` | Optional | Defaults to `chronoseek`. |

Optional Vidaio compression:

| Variable | Required when | Notes |
| --- | --- | --- |
| `VIDAIO_COMPRESSION_ENABLED=1` | Enabling Vidaio | Tries Vidaio before local ffmpeg compression. |
| `VIDAIO_API_BASE_URL` | `VIDAIO_COMPRESSION_ENABLED=1` | Vidaio compression API base URL. |
| `VIDAIO_API_KEY` | If required by Vidaio | Bearer token sent to Vidaio. |

Minimal `.env` shape:

```env
NETWORK=finney
NETUID=<netuid>
WALLET_NAME=<validator-coldkey>
HOTKEY_NAME=<validator-hotkey>
WALLET_PATH=~/.bittensor/wallets

CHUTES_API_KEY=<chutes-api-key>
HF_TOKEN=<hugging-face-token>

ENABLE_HARDENED_TASKS=1
VALIDATOR_TASK_SECRET=<long-random-secret>
HIPPIUS_S3_BUCKET=chronoseek
HIPPIUS_S3_ACCESS_KEY_ID=<hippius-access-key>
HIPPIUS_S3_SECRET_ACCESS_KEY=<hippius-secret-key>
```

## Run

Start the validator:

```bash
poetry run python validator.py
```

CLI flags can replace environment variables. For example:

```bash
poetry run python validator.py \
  --wallet.name <validator-coldkey> \
  --wallet.hotkey <validator-hotkey> \
  --netuid <netuid>
```

The validator checks that the hotkey is registered, loads permanent miner submissions from chain, disqualifies hotkeys with multiple revealed submissions, health-checks eligible Chutes runtimes, then enters the scoring loop.

## Hardened Task Flow

With `ENABLE_HARDENED_TASKS=1`, validators:

1. Sample an ActivityNet task internally.
2. Download the source video on the validator side.
3. Clip and re-encode the relevant window.
4. Optionally try Vidaio compression.
5. Fall back to local ffmpeg compression when Vidaio is disabled or unavailable.
6. Upload the final clip to Hippius S3.
7. Query miners with only the Hippius URL, query text, request ID, and `top_k=1`.
8. Score against clip-local ground truth intervals.

Hippius public URLs use this shape:

```text
https://s3.hippius.com/chronoseek/task-clips/<video-file-name-or-id>
```

At hardened validator startup, ChronoSeek checks that the configured Hippius bucket exists and applies a public-read bucket policy for `s3:GetObject` so miners can download uploaded task clips without Hippius credentials.

Uploaded Hippius videos are kept by default. Set `TASK_DELETE_REMOTE_ARTIFACTS=1` only if the validator should delete expired remote artifacts during cleanup.

## Video Download

Validators use the shared downloader for source videos. The downloader supports the same routing used by miner runtimes:

- Hippius URLs: Hippius S3-compatible API.
- YouTube, Vimeo, and other extractor-platform URLs: `yt-dlp`.
- Direct media URLs: raw HTTP.
- Failure path: fallback to the next supported strategy, then structured error details.

For restricted videos, configure cookies:

| Variable | Notes |
| --- | --- |
| `YTDLP_COOKIES` | Absolute path to a Netscape `cookies.txt` file. |
| `YTDLP_COOKIES_BROWSER` | Optional browser cookie source, for example `chrome:Default`, for local testing on machines where that browser profile is installed and readable. Empty by default. |
| `YTDLP_DENO_PATH` | Deno path used by yt-dlp challenge solving. |
| `YTDLP_NODE_PATH` | Optional Node.js path if you prefer Node for challenge solving. |

## Common Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `TASK_DATASET_PATH` | Empty | Local ActivityNet-style dataset path. If unset, Hugging Face is used. |
| `TASK_SPLIT` | `validation` | Dataset split. |
| `REQUIRE_ACCESSIBLE_VIDEOS` | `1` | Skip inaccessible source videos. |
| `TASK_MAX_SAMPLING_ATTEMPTS` | `50` | Attempts to find an accessible task. |
| `TASK_CLIP_CACHE_DIR` | `~/.cache/chronoseek/task-clips` | Local generated clip cache. |
| `TASK_ARTIFACT_MANIFEST_PATH` | Empty | Local uploaded artifact manifest path. |
| `TASK_ARTIFACT_PREFIX` | `task-clips` | Hippius object-key prefix. |
| `TASK_CLIP_TTL_HOURS` | `6` | Local manifest/cache expiry. |
| `TASK_DELETE_REMOTE_ARTIFACTS` | `0` | Remote Hippius deletion is opt-in. |
| `TASK_VIDEO_COOLDOWN_HOURS` | `24` | Cooldown before reusing a source video. |
| `TASK_CAPTION_COOLDOWN_HOURS` | `168` | Cooldown before reusing a source caption. |
| `TASK_MIN_CLIP_DURATION_SECONDS` | `30` | Minimum generated clip duration. |
| `TASK_MAX_CLIP_DURATION_SECONDS` | `180` | Maximum generated clip duration. |
| `TASK_SOURCE_DOWNLOAD_TIMEOUT_SECONDS` | `120` | Source download timeout for clipping. |
| `VALIDATOR_EVAL_TOP_K` | `1` | Number of miner results requested and scored. |
| `TASK_MAX_PREDICTION_DURATION_SECONDS` | `60` | Maximum scored prediction duration unless ground truth is longer. |
| `MINER_REQUEST_TIMEOUT_SECONDS` | `150` | Per-miner `/search` timeout. |
| `MINER_SUBMISSION_CACHE_TTL_SECONDS` | `300` | Chain submission cache TTL. |
| `MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS` | `60` | Miner submission refresh interval. |
| `MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS` | `10` | Per-runtime `/health` timeout. |
| `MINER_EMISSION_BURN_PERCENT` | `0` | Percent of emissions assigned to UID 0 burn. |

## Hippius Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `HIPPIUS_S3_ENDPOINT_URL` | `https://s3.hippius.com` | Hippius S3-compatible endpoint. |
| `HIPPIUS_S3_PUBLIC_BASE_URL` | `https://s3.hippius.com` | Public base URL for miner-facing clips. |
| `HIPPIUS_S3_BUCKET` | `chronoseek` | Bucket for validator task clips. |
| `HIPPIUS_S3_REGION` | `decentralized` | S3 signing region. |
| `HIPPIUS_S3_TIMEOUT_SECONDS` | `60` | Upload/download/delete timeout. |

## Vidaio Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `VIDAIO_COMPRESSION_ENABLED` | `0` | Try Vidaio before local ffmpeg compression. |
| `VIDAIO_API_BASE_URL` | Empty | Vidaio compression API base URL. |
| `VIDAIO_API_KEY` | Empty | Optional bearer token. |
| `VIDAIO_TIMEOUT_SECONDS` | `60` | Vidaio request timeout. |

## Operational Notes

- Do not send original ActivityNet source URLs, source video IDs, raw captions, or ground-truth timestamps to miners.
- Keep `VALIDATOR_TASK_SECRET`, Hippius credentials, Chutes credentials, and Vidaio credentials out of committed files.
- Hardened task generation increases validator-side CPU, disk, network, and storage usage.
- See [Owner and development guide](./development.md) for live module checks.
