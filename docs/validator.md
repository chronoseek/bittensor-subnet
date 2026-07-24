# Validator Guide

This guide is for operators running a ChronoSeek validator.

Validators generate synthetic video-search tasks, query miner Chutes runtimes, score responses, and set on-chain weights. In hardened mode, validators also create private task clips and send miners only a compressed clipped task URL plus query. The final URL comes from Vidaio when remote compression succeeds, or from validator-managed Hippius upload after local ffmpeg fallback.

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
| `NETWORK` | No | Defaults to `finney`; set it explicitly for `test`, `local`, or another network. |
| `NETUID` | No | Defaults to the ChronoSeek mainnet subnet, `20`. |
| `WALLET_PATH` | If non-default | Defaults to `~/.bittensor/wallets`. |
| `ENFORCE_ONE_HOTKEY_ONE_SUBMISSION` | No | Defaults to `1`. Set to `0` only for tests that require replaceable miner submissions; validators then use the newest revealed submission and do not apply the duplicate-submission penalty. |
| `CHUTES_API_KEY` | Yes | Used to query private Chutes runtimes. |
| `HF_TOKEN` | Yes, unless `--task-dataset-path` is passed | Required for Hugging Face ActivityNet loading and model downloads. |

Hardened task generation:

| Setting | Required when | Notes |
| --- | --- | --- |
| `DEFAULT_ENABLE_HARDENED_TASKS` | Default | Hardened task generation is enabled by default. Use `--disable-hardened-tasks` only for bootstrap/debug runs. |
| `VALIDATOR_TASK_SECRET` or `--validator-task-secret` | Normal validator operation | Set a long random secret. Used for private deterministic sampling. |
| `HIPPIUS_S3_ACCESS_KEY_ID` | Normal validator operation | Hippius S3 access key ID. |
| `HIPPIUS_S3_SECRET_ACCESS_KEY` | Normal validator operation | Hippius S3 secret key. |
| `HIPPIUS_S3_BUCKET` or `--hippius-s3-bucket` | Normal validator operation | Defaults to `chronoseek`, but that name may already be owned by another operator. Set this to a bucket your Hippius credentials own or can create — startup calls `set_bucket_policy` on it and fails with `AccessDenied` otherwise. |
| `DEFAULT_HIPPIUS_S3_*` constants | Non-default storage only | Update endpoint, public base URL, or region constants when a deployment uses different Hippius-compatible storage. |

Vidaio compression:

| Setting | Required when | Notes |
| --- | --- | --- |
| `VIDAIO_COMPRESSION_ENABLED`, `DEFAULT_VIDAIO_COMPRESSION_ENABLED`, or `--disable-vidaio-compression` | Normal validator operation | Enabled by default. Set `VIDAIO_COMPRESSION_ENABLED=0` or pass `--disable-vidaio-compression` to use local ffmpeg directly. |
| `DEFAULT_VIDAIO_API_BASE_URL` or `--vidaio-api-base-url` | Non-default Vidaio endpoint only | Defaults to `https://api.vidaio.io`; not configured through `.env`. |
| `VIDAIO_API_KEY` | Vidaio protected endpoints | API key sent as the `X-API-Key` header. |
| `VIDAIO_POLL_INTERVAL_SECONDS` or `--vidaio-poll-interval-seconds` | Vidaio workflow polling | Defaults to `15` seconds to stay below the 6-read/minute workflow-read limit. |

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
NETUID=20
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

### Docker deployment with automatic updates

The GitHub Actions workflow publishes the validator image to
`creativeessence/chronoseek-subnet` by default. Configure the repository under
**Settings → Secrets and variables → Actions**:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `DOCKERHUB_USERNAME` | Docker Hub user allowed to push the image. |
| Secret | `DOCKERHUB_TOKEN` | Docker Hub access token with read/write permission. Do not use the account password. |
| Variable (optional) | `DOCKERHUB_IMAGE` | Alternate `namespace/repository`, if not using `creativeessence/chronoseek-subnet`. |

Create the matching Docker Hub repository before running the workflow. A public
repository is the simplest deployment because validator hosts and Watchtower
can pull without registry credentials.

The two GitHub secrets above are only for publishing images. Do not put
validator runtime credentials in GitHub Actions. Each validator operator keeps
their own runtime credentials on their validator host.

On each validator host, create and protect the runtime environment file:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` with that validator's settings:

```env
DOCKER_IMAGE=creativeessence/chronoseek-subnet:latest
NETWORK=finney
NETUID=20

WALLET_NAME=<validator-coldkey>
HOTKEY_NAME=<validator-hotkey>
WALLET_PATH=~/.bittensor/wallets

CHUTES_API_KEY=<validator-chutes-api-key>
HF_TOKEN=<validator-hugging-face-token>
VALIDATOR_TASK_SECRET=<long-random-validator-secret>

HIPPIUS_S3_BUCKET=<validator-owned-bucket>
HIPPIUS_S3_ACCESS_KEY_ID=<hippius-access-key>
HIPPIUS_S3_SECRET_ACCESS_KEY=<hippius-secret-key>

VIDAIO_COMPRESSION_ENABLED=1
VIDAIO_API_KEY=<vidaio-api-key-if-required>
LOG_LEVEL=INFO
```

Compose injects every entry from `.env` into the validator container. The
wallet keys themselves are not environment variables: Compose mounts
`~/.bittensor/wallets` read-only, and `WALLET_NAME`/`HOTKEY_NAME` select the
wallet files to use. The `.env` file is ignored by both Git and Docker builds,
so it is neither committed nor baked into the published image.

Then start the deployment:

```bash
docker compose pull
docker compose up -d
docker compose logs -f chronoseek-validator
```

After changing an environment value, recreate the validator container:

```bash
docker compose up -d --force-recreate validator
```

Avoid posting the output of `docker compose config` in logs or support chats;
depending on the Compose version, it may include resolved secret values.

Compose pulls the published image at startup. Watchtower checks Docker Hub every
five minutes and restarts the validator when a new `latest` image is available.
The workflow publishes on pushes to `main` or `master`, on `v*` release tags,
and through manual workflow dispatch.

CLI flags can replace environment variables and override code defaults. For example:

```bash
poetry run python validator.py \
  --wallet.name <validator-coldkey> \
  --wallet.hotkey <validator-hotkey> \
  --netuid <netuid>
```

The validator checks that the hotkey is registered, loads miner submissions from chain, health-checks eligible Chutes runtimes, then enters the scoring loop. By default, hotkeys with multiple revealed submissions are disqualified. With `ENFORCE_ONE_HOTKEY_ONE_SUBMISSION=0`, the newest revealed submission is used instead.

For a local preflight that does not query miners or set weights, generate one hardened task with the bundled smoke manifest:

```bash
poetry run python scripts/verify_hardened_task_generation.py --use-smoke-dataset --no-require-accessible-videos
```

## Hardened Task Flow

With `DEFAULT_ENABLE_HARDENED_TASKS=True`, validators:

1. Sample an ActivityNet task internally.
2. Download the source video on the validator side.
3. Clip and re-encode the relevant window.
4. If enabled, upload a temporary public source clip and try Vidaio compression.
5. When Vidaio succeeds, use the compressed video URL returned by Vidaio.
6. Fall back to local ffmpeg compression and upload that final clip to Hippius S3 when Vidaio is disabled or unavailable.
7. Query miners with only the miner-facing compressed clip URL, query text, request ID, and `top_k=1`.
8. Score against clip-local ground truth intervals.

Hippius public URLs use this shape:

```text
https://s3.hippius.com/chronoseek/task-clips/<video-file-name-or-id>
```

At hardened validator startup, ChronoSeek checks that the configured Hippius bucket exists and applies a public-read bucket policy for `s3:GetObject` so Vidaio can fetch temporary source clips and miners can download locally compressed fallback task clips without Hippius credentials.

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
