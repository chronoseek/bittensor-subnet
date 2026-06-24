# Validator Guide

This guide is for operators running a ChronoSeek validator.

Validators generate synthetic video-search tasks, query miner Chutes runtimes, score responses, and set on-chain weights. In hardened mode, validators also create private task clips, upload them to Hippius, and send miners only the clipped task URL plus query.

## Prerequisites

- Python 3.12+
- Poetry
- A registered validator hotkey on the configured subnet
- `ffmpeg` available in `PATH` when hardened task clipping/compression is enabled
- A Hugging Face token, unless `--task-dataset-path` points to a local ActivityNet-style dataset
- A Chutes API key for querying private miner runtimes
- Hippius S3 credentials when hardened task generation is enabled

Install dependencies from the repository root:

```bash
poetry install
cp .env.example .env
```

`.env.example` intentionally stays small. Normal validators should only need identity, credentials, secrets, and host-specific access files in `.env`; runtime defaults live as `DEFAULT_*` constants in `chronoseek/constants.py` and can be overridden with CLI flags for experiments.

## Environment Variables

Required for normal validator operation:

| Variable | Required | Notes |
| --- | --- | --- |
| `WALLET_NAME` | Yes | Coldkey wallet name. |
| `HOTKEY_NAME` | Yes | Registered validator hotkey name. |
| `NETWORK` | Yes | Usually `finney`, `test`, or `local`. |
| `NETUID` | Yes | ChronoSeek subnet netuid. |
| `WALLET_PATH` | If non-default | Defaults to `~/.bittensor/wallets`. |
| `CHUTES_API_KEY` | Yes | Used to query private Chutes runtimes. |
| `HF_TOKEN` | Yes, unless `--task-dataset-path` is passed | Required for Hugging Face ActivityNet loading and model downloads. |

Hardened task generation:

| Setting | Required when | Notes |
| --- | --- | --- |
| `DEFAULT_ENABLE_HARDENED_TASKS` | Default | Hardened task generation is enabled by default. Use `--disable-hardened-tasks` only for bootstrap/debug runs. |
| `VALIDATOR_TASK_SECRET` or `--validator-task-secret` | Normal validator operation | Set a long random secret. Used for private deterministic sampling. |
| `HIPPIUS_S3_ACCESS_KEY_ID` | Normal validator operation | Hippius S3 access key ID. |
| `HIPPIUS_S3_SECRET_ACCESS_KEY` | Normal validator operation | Hippius S3 secret key. |
| `DEFAULT_HIPPIUS_S3_*` constants | Non-default storage only | Update endpoint, public base URL, bucket, or region constants when a deployment uses different Hippius-compatible storage. |

Optional Vidaio compression:

| Variable | Required when | Notes |
| --- | --- | --- |
| `--vidaio-compression-enabled` or `DEFAULT_VIDAIO_COMPRESSION_ENABLED` | Enabling Vidaio | Disabled by default because no public Vidaio API is available yet. When enabled, tries Vidaio before local ffmpeg compression. |
| `VIDAIO_API_BASE_URL` | Vidaio is enabled | Vidaio compression API base URL. |
| `VIDAIO_API_KEY` | If required by Vidaio | Bearer token sent to Vidaio. |

Optional shared validator/miner video access:

| Variable | Notes |
| --- | --- |
| `YTDLP_COOKIES` | Absolute path to a Netscape `cookies.txt` file. Validators use this when downloading source videos for task generation; miner Chutes image builds copy a readable cookies file into the runtime image. |
| `YTDLP_COOKIES_BROWSER` | Optional browser cookie source, for example `chrome:Default`, for local testing on machines where that browser profile is installed and readable. |
| `YTDLP_DENO_PATH` | Optional Deno path for yt-dlp JavaScript challenge solving when the executable is not discoverable automatically. |
| `YTDLP_NODE_PATH` | Optional Node.js path for yt-dlp JavaScript challenge solving. |

Minimal `.env` shape:

```env
NETWORK=finney
NETUID=<netuid>
WALLET_NAME=<validator-coldkey>
HOTKEY_NAME=<validator-hotkey>
WALLET_PATH=~/.bittensor/wallets

CHUTES_API_KEY=<chutes-api-key>
HF_TOKEN=<hugging-face-token>

VALIDATOR_TASK_SECRET=<long-random-secret>
HIPPIUS_S3_ACCESS_KEY_ID=<hippius-access-key>
HIPPIUS_S3_SECRET_ACCESS_KEY=<hippius-secret-key>
```

Advanced runtime defaults such as dataset path, hardened-mode enablement, clip TTL, miner request timeout, eval `top_k`, Chutes base domain, and Hippius endpoint/bucket values are no longer expected in `.env`. Use the code constants in `chronoseek/constants.py` for subnet-wide defaults, or pass the corresponding CLI flag for a one-off run.

## Run

Start the validator:

```bash
poetry run python validator.py
```

CLI flags can replace environment variables and override code defaults. For example:

```bash
poetry run python validator.py \
  --wallet.name <validator-coldkey> \
  --wallet.hotkey <validator-hotkey> \
  --netuid <netuid>
```

The validator checks that the hotkey is registered, loads permanent miner submissions from chain, disqualifies hotkeys with multiple revealed submissions, health-checks eligible Chutes runtimes, then enters the scoring loop.

For a local preflight that does not query miners or set weights, generate one hardened task with the bundled smoke manifest:

```bash
poetry run python scripts/verify_hardened_task_generation.py --use-smoke-dataset --no-require-accessible-videos
```

## Hardened Task Flow

With `DEFAULT_ENABLE_HARDENED_TASKS=True`, validators:

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

Uploaded Hippius videos are kept by default. Use `--task-delete-remote-artifacts` only if the validator should delete expired remote artifacts during cleanup.

## Video Download

Validators use the shared downloader for source videos. The downloader supports the same routing used by miner runtimes:

- Hippius URLs: Hippius S3-compatible API.
- YouTube, Vimeo, and other extractor-platform URLs: `yt-dlp`.
- Direct media URLs: raw HTTP.
- Failure path: fallback to the next supported strategy, then structured error details.

For restricted videos, configure `YTDLP_COOKIES` or `YTDLP_COOKIES_BROWSER` in `.env`. Validators use this downloader while generating tasks, and miner runtimes use the same settings when fetching validator-provided video URLs.

## Defaults And Overrides

ChronoSeek keeps ordinary tuning in `DEFAULT_*` constants in `chronoseek/constants.py`. This includes:

- task sampling, split, accessibility, cooldown, and cache defaults
- hardened clip duration, TTL, cleanup, encoding, canary, and exposure defaults
- Hippius endpoint, public base URL, bucket, region, and timeout defaults
- miner request, submission refresh, health-check, scoring, latency, and burn defaults

For one-off local testing, use CLI flags instead of adding new `.env` entries:

```bash
poetry run python validator.py \
  --task-clip-ttl-hours 2 \
  --miner-request-timeout-seconds 90 \
  --hippius-s3-bucket chronoseek-test
```

## Operational Notes

- Do not send original ActivityNet source URLs, source video IDs, raw captions, or ground-truth timestamps to miners.
- Keep `VALIDATOR_TASK_SECRET`, Hippius credentials, Chutes credentials, and Vidaio credentials out of committed files.
- Hardened task generation increases validator-side CPU, disk, network, and storage usage.
- See [Owner and development guide](./development.md) for live module checks.

## Interval Scoring

Validator scoring is still accuracy-first and protocol-compatible, but the
default score is now shape-aware interval quality rather than raw IoU alone.
The best valid top-1 prediction/ground-truth pair is scored by IoU, then
dampened for sloppy interval shape:

- center alignment: prediction center should land near the target moment center
- boundary alignment: start and end should closely match the ground truth
- duration alignment: broad windows are penalized even when they overlap

Malformed intervals, empty responses, timeouts, out-of-clip intervals, and
oversized intervals still score zero. Extra results beyond the validator's
configured top-k do not improve the score.

When the latency multiplier is enabled, the interval quality score is multiplied
by a gentle latency factor. The multiplier is `1.0` through the configured
grace period, then decays toward the configured minimum near the miner request
timeout. A zero-quality response remains zero regardless of latency.

## Adversarial Task Transforms

Hardened task generation can include same-video hard-negative moments in the
clip when another annotated moment fits inside the configured maximum duration.
The generator also records the selected encoding transform profile and context
padding metadata on the internal `ValidationTask`. Miners still receive only
the artifact URL, query, request ID, protocol version, and `top_k`.

## Canary Tasks

Canaries are internal validator probes sampled from hardened tasks at the
configured canary task rate. The initial canary types are:

- `absent`: replaces the query with an unlikely absent event, clears ground
  truth, and expects an empty successful response.
- `hard-negative`: tags clips containing same-video distractor moments.
- `repeated`: tags tasks with multiple valid target intervals.

Timeouts, protocol errors, and failed requests are never rewarded as absent
canaries; only a successful empty response can receive the absent-canary score.

## Task Exposure Controls

The artifact manifest records source video, source caption, query variant,
transform profile, crop bucket, and task family for active generated clips.
Validators can set `MAX_ACTIVE_TASKS_PER_*` limits to reduce repeated exposure
while task artifacts remain active. The local hardened-task verifier prints an
exposure summary so operators can inspect active task-bank freshness.

## Telemetry

Validators keep report-only miner behavior telemetry in memory and can write it
with `--validator-telemetry-path`. The snapshot includes per-miner score,
latency, failure, timeout, task-family, canary, hard-negative, and
repeated-duration signals. Suspicion flags are logged for visibility but do not
affect weights until an aggregation policy explicitly consumes them.

## Score Aggregation

The validator now computes explicit quality, reliability, consistency, and
suspicion components before updating moving scores. By default only the quality
EMA is active, matching the previous `alpha = 0.1` behavior. Setting
The reliability, consistency, and suspicion weights are configured by constants
or CLI flags. Setting any of them above zero turns on post-quality multipliers
based on the telemetry summary for each miner.
