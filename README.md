# ChronoSeek

> **Google for Videos.**

ChronoSeek uses a Chutes-backed `v2.0` `Eval/Serve Split` architecture.

The full design note is [ChronoSeek v2.0 Eval/Serve Split](./docs/CHRONOSEEK_V2_EVAL_SERVE_SPLIT.md).

## Version Guide

### `v1.x` Earlier testnet design

- miners ran live search services
- validators generate synthetic tasks and score miner responses
- organic traffic could be forwarded through validator-owned infrastructure

### `v2.0` Target design: `Eval/Serve Split`

- miners deploy private retrieval runtimes to Chutes
- miners commit structured deployment metadata on-chain
- validators read the latest valid miner submission per hotkey and query the private Chutes directly
- synthetic evaluation remains the only scoring source
- the public API is no longer intended to forward organic requests to miners
- neither miners nor validators publish ports through subnet metadata
- subnet admins can promote selected winning private Chutes into immutable public clones for owner-run API serving
- promoted clones are locked to the exact Docker image that was running at clone time

## Current Scope

### Current subnet capabilities

- multimodal baseline with visual retrieval plus transcript-based speech understanding
- deterministic validator evaluation using ActivityNet Captions
- IoU-based scoring with moving-average weight updates
- chain-only miner submission and validator discovery

### `v2.0` architecture changes in progress

1. Replace miner-facing organic routing with owner-controlled serving.
2. Treat miner submissions as hosted retrieval runtimes, not only model weights.
3. Resolve miner deployment metadata from chain for search routing.
4. Use Chutes promotion and immutable Docker-image cloning for production-facing API backends.

## Documentation

- [Problem Statement](docs/PROBLEM_STATEMENT.md)
- [System Design](docs/DESIGN.md)
- [Business Logic](docs/BUSINESS_LOGIC.md)

## Core Concept

ChronoSeek is a decentralized protocol where:

- **Miners** compete by improving semantic video retrieval runtimes.
- **Validators** score miners on synthetic tasks and set on-chain weights.
- **The owner-run API** serves organic traffic from operator-controlled deployments rather than volatile miner endpoints.

## Architecture Overview

### Historical `v1.x`

```text
User / Client
   │
   ▼
Validator process
   ├─ Deterministic evaluation
   └─ Organic / developer query routing (removed in v2.0)
   │
   ▼
Miners
   └─ Semantic video analysis
```

### `v2.0`

```text
Synthetic path
Validator
  -> chain submission metadata
  -> private miner Chutes
  -> scoring and weights

Organic path
User / Developer
  -> owner-run API
  -> promoted immutable Chutes clones
  -> locked Docker-image runtime
  -> stable response
```

## Installation & Setup

This project uses `poetry` for dependency management.

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)

### Clone & install

```bash
git clone https://github.com/chronoseek/bittensor-subnet.git
cd bittensor-subnet
poetry install
```

### Activate the virtual environment

```bash
poetry env activate
```

### Setting up environment variables

Before running the project, copy the example environment file and edit it with your credentials:

```bash
cp .env.example .env
```

Open the new `.env` file and set all required variables, including your Hugging Face token (`HF_TOKEN`) and Chutes API key (`CHUTES_API_KEY`). Refer to comments in `.env.example` for guidance on which variables need to be filled in for your use case.

## Runtime Notes

### Reference retrieval runtime

The reference runtime currently:

- downloads source video
- performs CLIP-based retrieval
- optionally extracts and transcribes speech
- returns ranked timestamp intervals
- is intended to run inside the submitted Chutes runtime, not as a locally served subnet miner

### `v2.0` miner submission model

The miner submission unit is a Chutes-hosted retrieval runtime with immutable deployment metadata. In practice this means:

- miners still own training and deployment
- miner submissions may include full custom runtime code for video access, clip traversal, transcription, retrieval, ranking, and formatting
- validators still own scoring
- the product API no longer depends on direct miner fanout
- promoted serving uses a Chutes clone locked to the exact Docker image that was running at clone time

Miners participate by deploying their runtime to Chutes, then committing a validated v2 submission payload:

```bash
cp chronoseek_chute.example.py chronoseek_chute.py
poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee

poetry run python miner.py \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --chute-slug <username>-chronoseek-runtime-<timestamp>
```

The helper boundaries are intentionally narrow:

- `chronoseek.chain.submissions` owns on-chain miner metadata commit/fetch.
- `chronoseek.chutes.deployment` owns Chutes API build/deploy calls and metadata extraction from SDK-defined objects.
- `chronoseek.chutes.runtime` owns Chutes endpoint resolution, auth headers, and runtime health checks.
- validator scoring remains in the v1.x evaluation loop; v2.0 only changes how validators discover and reach miners.

## Running Nodes

### Deploy a miner Chutes runtime

Use the deployment wrapper to publish a Chutes runtime through Chutes APIs. The wrapper does not commit anything on-chain; it prints normalized deployment metadata and the exact `miner.py` command to run next.

The Chute is still defined with the Chutes SDK, so the wrapper expects a module reference, not a file path:

```text
chronoseek_chute:chute
```

This means: load the root-level `chronoseek_chute.py`, then use the object named `chute`. Create your local copy from the committed example:

```bash
cp chronoseek_chute.example.py chronoseek_chute.py
```

Then edit `chronoseek_chute.py` before deploying:

- `CHUTES_ACCOUNT`: account namespace required by the Chutes SDK object construction. This is not part of ChronoSeek miner identity or the runtime slug.
- `CHUTE_BASE_NAME`: the shared base name, `chronoseek-runtime`. The deploy wrapper renders the Chutes API name with brand casing as `ChronoSeek-runtime-<timestamp>` because Chutes derives the public slug from `username + name`.
- `CHRONOSEEK_LOGO_URL`: card/image avatar uploaded to Chutes before build/deploy. The default is `https://chronoseek.org/logo.png`.
- `RUNTIME_REVISION`: git SHA, image version, or other immutable build label.
- `CHRONOSEEK_PACKAGE`: where Chutes can install your runtime code from. This can be a public git URL, private git URL with deploy credentials, or your own package/image install command.
- `node_selector`: GPU requirements, especially `gpu_count` and `min_vram_gb_per_gpu`.
- `concurrency`: keep this low at first because video retrieval is GPU/CPU/memory heavy.
- `allow_external_egress=True`: required because the runtime must fetch validator task videos.

The template exposes `/health` and `/search` as native Chutes SDK cords. It does not start a local FastAPI server inside Chutes.

### Local Chutes runtime testing

Use local Chutes before production deployment. Production build/deploy calls consume Chutes credits; only the project owner should run them intentionally.

Print the exact local build/run commands for the current `chronoseek_chute.py`:

```bash
poetry run python scripts/test_chutes_runtime_local.py --print-commands
```

Build the image locally:

```bash
poetry run python scripts/test_chutes_runtime_local.py --build
```

Run the local Chutes container in a separate terminal:

```bash
poetry run python scripts/test_chutes_runtime_local.py --run
```

Smoke-test the already-running local runtime:

```bash
poetry run python scripts/test_chutes_runtime_local.py --smoke
```

For a cheap readiness check that avoids video inference:

```bash
poetry run python scripts/test_chutes_runtime_local.py --smoke --health-only
```

This helper follows the official local Chutes flow (`chutes build <ref> --local`, then `docker run ... chutes run <ref> --dev`) and never calls the production Chutes API.

### Production Chutes deployment

Build and deploy from the SDK-defined object through the Chutes APIs:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee \
  --artifact-id chronoseek-runtime
```

The deploy wrapper is an off-chain Chutes helper. It does not load wallets, verify metagraph registration, or touch the chain. Every run generates a UTC millisecond timestamp for runtime uniqueness. With Chutes account `chronoseek` and timestamp `20260510143015999`, the Chutes API `name` is `ChronoSeek-runtime-20260510143015999`, the human display label in logs/readme is `ChronoSeek Runtime`, and the routable `slug` is `chronoseek-chronoseek-runtime-20260510143015999`. Validators resolve that to `https://chronoseek-chronoseek-runtime-20260510143015999.chutes.ai`. Use the exact slug printed by the deploy helper when committing metadata with `miner.py`.

Before build/deploy, the wrapper downloads `CHRONOSEEK_LOGO_URL`, uploads it to Chutes, and sends the returned `logo_id` with both the image build and chute deploy payloads. During image build, it also checks `YTDLP_COOKIES`; when it points to an existing local cookies file, the wrapper copies it into the build context, adds it under `/opt/chronoseek/miner-files/ytdlp/...`, and rewrites the cookie env var inside the image to that container path. The example chute installs Deno at `/opt/deno/bin/deno`, sets the yt-dlp Deno env var, defaults the browser cookie source to `chrome:Default`, and forwards `HF_TOKEN` into the image when it is present in the local environment.

Set `RUNTIME_REVISION` inside `chronoseek_chute.py` for the actual Chutes image/chute revision. The helper's `--revision` flag only overrides the on-chain provenance value printed for `miner.py`.

Before building, the wrapper checks whether the generated Chutes image already exists. If it does, it prompts with `[Y/n]`; pressing Enter or typing `y` deletes the existing image and submits a new build, while `n` aborts. For non-interactive rebuilds, pass `--overwrite-existing-image`; to fail fast without prompting, pass `--no-overwrite-existing-image`.

If the image is already built, deploy only:

```bash
poetry run python scripts/deploy_chutes_runtime.py --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee
```

If you only need to print the suggested on-chain commit command from the SDK-defined object:

```bash
poetry run python scripts/deploy_chutes_runtime.py \
  --chute-ref chronoseek_chute:chute
```

For custom ChronoSeek runtimes, no Hugging Face repo is required. The Chutes deployment input is the Python Chute SDK definition, converted into Chutes API requests by `scripts/deploy_chutes_runtime.py`. `--artifact-id`, `--artifact-revision`, and `--artifact-digest` are optional provenance fields for your runtime image/build, not model repository requirements.

### Commit a miner runtime submission

```bash
poetry run python miner.py \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --chute-id chute-deployment-id \
  --chute-slug <username>-chronoseek-runtime-<timestamp> \
  --artifact-revision immutable-revision
```

### Chutes runtime entrypoint

The runtime is deployed from the Chutes SDK definition through Chutes APIs and exposes `/health` and `/search` as native Chutes cords. The subnet miner command does not serve HTTP locally.

### Start the validator

```bash
poetry run python validator.py
```

The validator reads latest revealed metadata for hotkeys present in the metagraph and queries the resolved private Chutes runtime endpoints. There is no local validator HTTP service or local metadata-file fallback.

## Environment Variables

The subnet now supports `v2.0` submission-based evaluation. Serving promotion and owner-run public API selection live outside this subnet process.

### Chain

| Variable  | Description                         | Default  |
| --------- | ----------------------------------- | -------- |
| `NETWORK` | Network (`finney`, `test`, `local`) | `finney` |
| `NETUID`  | Subnet NetUID                       | `1`      |

### Wallet

| Variable      | Description                 | Default                |
| ------------- | --------------------------- | ---------------------- |
| `WALLET_NAME` | Name of your coldkey        | `default`              |
| `HOTKEY_NAME` | Name of your hotkey         | `default`              |
| `WALLET_PATH` | Path to your wallet storage | `~/.bittensor/wallets` |

### Chutes

| Variable                     | Description                                                                | Default                                         |
| ---------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `CHUTES_API_KEY`             | Chutes API token for build/deploy and private eval                         | ``                                              |
| `CHUTES_BASE_DOMAIN`         | Domain used to resolve `chute_slug` submissions                            | `chutes.ai`                                     |
| `MIN_VALIDATOR_STAKE`        | Minimum validator stake required by the Chutes runtime                     | `10000`                                         |

### Miner Video Download

| Variable                             | Description                                                                 | Default              |
| ------------------------------------ | --------------------------------------------------------------------------- | -------------------- |
| `YTDLP_COOKIES`           | Optional local Netscape `cookies.txt`; copied into the Chutes image at build | ``                   |
| `YTDLP_COOKIES_BROWSER`   | Browser cookie source used if no cookies file is available                  | `chrome:Default`     |
| `YTDLP_NODE_PATH`         | Optional Node.js runtime path for yt-dlp EJS challenge solver               | ``                   |
| `YTDLP_DENO_PATH`         | Deno runtime path for yt-dlp EJS challenge solver                           | `/opt/deno/bin/deno` |

### Hugging Face

| Variable                  | Description                                                | Default                |
| ------------------------- | ---------------------------------------------------------- | ---------------------- |
| `HF_TOKEN`                | Hugging Face token                                         | `None`                 |
| `HF_HOME`                 | Hugging Face cache directory                               | `~/.cache/huggingface` |
| `HF_ACTIVITYNET_FILENAME` | Optional filename override inside the ActivityNet snapshot | ``                     |

### Validator Task Generation

| Variable                     | Description                                    | Default      |
| ---------------------------- | ---------------------------------------------- | ------------ |
| `TASK_DATASET_PATH`          | Optional local validator dataset path          | ``           |
| `TASK_SPLIT`                 | Validator task split                           | `validation` |
| `REQUIRE_ACCESSIBLE_VIDEOS`  | Skip inaccessible validator task videos        | `1`          |
| `TASK_MAX_SAMPLING_ATTEMPTS` | Max tries to find an accessible validator task | `50`         |

### Validator Video Availability Cache

| Variable                             | Description                                          | Default |
| ------------------------------------ | ---------------------------------------------------- | ------- |
| `VIDEO_AVAILABILITY_CACHE_PATH`      | Legacy base path for validator availability caches   | ``      |
| `ACCESSIBLE_VIDEO_CACHE_PATH`        | JSON cache path for accessible validator videos      | ``      |
| `INACCESSIBLE_VIDEO_CACHE_PATH`      | JSON cache path for inaccessible validator videos    | ``      |
| `VIDEO_AVAILABILITY_CACHE_TTL_HOURS` | TTL for cached video availability checks             | `24`    |
| `VIDEO_AVAILABILITY_TIMEOUT`         | Timeout for validator-side video availability checks | `20`    |

### Validator Evaluation

| Variable                                    | Description                                                | Default |
| ------------------------------------------- | ---------------------------------------------------------- | ------- |
| `MINER_REQUEST_TIMEOUT_SECONDS`             | Per-miner timeout for validator runtime search requests    | `150`   |
| `MINER_SUBMISSION_CACHE_TTL_SECONDS`        | Cache TTL for loaded v2 submissions                        | `300`   |
| `MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS` | Interval between validator submission metadata refreshes   | `60`    |
| `MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS`   | Per-runtime timeout for responsive miner `/health` checks  | `10`    |
| `MINER_EMISSION_BURN_PERCENT`               | Percent of miner emissions to burn through UID 0 weighting | `0`     |

### Logging

| Variable    | Description       | Default |
| ----------- | ----------------- | ------- |
| `LOG_LEVEL` | Logging verbosity | `INFO`  |
