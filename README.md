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

### Hugging Face token

```bash
export HF_TOKEN=your_token_here
```

## Runtime Notes

### Reference retrieval runtime

The reference runtime currently:

- downloads source video
- performs CLIP-based retrieval
- optionally extracts and transcribes speech
- returns ranked timestamp intervals
- is intended to run inside the submitted Chutes runtime, not as a locally served subnet miner

### `v2.0` miner submission model

The miner submission unit is shifting toward a Chutes-hosted retrieval runtime with immutable deployment metadata. In practice this means:

- miners still own training and deployment
- miner submissions may include full custom runtime code for video access, clip traversal, transcription, retrieval, ranking, and formatting
- validators still own scoring
- the product API no longer depends on direct miner fanout
- promoted serving uses a Chutes clone locked to the exact Docker image that was running at clone time

Miners participate by deploying their runtime to Chutes and committing a validated v2 submission payload:

```bash
poetry run python miner.py \
  --chute-id chute-deployment-id \
  --chute-slug chronoseek-runtime \
  --artifact-revision immutable-revision
```

## Running Nodes

### Commit a miner runtime submission

```bash
poetry run python miner.py \
  --chute-id chute-deployment-id \
  --chute-slug chronoseek-runtime \
  --artifact-revision immutable-revision
```

### Chutes runtime entrypoint

```bash
uvicorn chronoseek.miner.runtime:app --host 0.0.0.0 --port 8000
```

The runtime is deployed on Chutes and exposes `/health` and `/search`. The subnet miner command does not serve HTTP locally.

### Start the validator

```bash
poetry run python validator.py
```

The validator reads latest revealed metadata for hotkeys present in the metagraph and queries the resolved private Chutes runtime endpoints. There is no local validator HTTP service or local metadata-file fallback.

## Environment Variables

The subnet now supports `v2.0` submission-based evaluation. Serving promotion and owner-run public API selection live outside this subnet process.

| Variable                                     | Description                                                   | Default                 |
| -------------------------------------------- | ------------------------------------------------------------- | ----------------------- |
| `WALLET_NAME`                                | Name of your coldkey                                          | `default`               |
| `HOTKEY_NAME`                                | Name of your hotkey                                           | `default`               |
| `WALLET_PATH`                                | Path to your wallet storage                                   | `~/.bittensor/wallets/` |
| `NETUID`                                     | Subnet NetUID                                                 | `298`                   |
| `NETWORK`                                    | Network (`finney`, `test`, `local`)                           | `test`                  |
| `MIN_VALIDATOR_STAKE`                        | Minimum validator stake required by the Chutes runtime        | `10000`                 |
| `CHRONOSEEK_YTDLP_COOKIES`                   | Optional path to Netscape `cookies.txt` for YouTube auth      | ``                      |
| `CHRONOSEEK_YTDLP_COOKIES_BROWSER`           | Optional browser source for cookies                           | ``                      |
| `CHRONOSEEK_YTDLP_NODE_PATH`                 | Optional Node.js runtime path for yt-dlp EJS challenge solver | ``                      |
| `CHRONOSEEK_YTDLP_DENO_PATH`                 | Optional Deno runtime path for yt-dlp EJS challenge solver    | ``                      |
| `LOG_LEVEL`                                  | Logging verbosity                                             | `INFO`                  |
| `HF_TOKEN`                                   | Hugging Face token                                            | `None`                  |
| `HF_HOME`                                    | Hugging Face cache directory                                  | `~/.cache/huggingface`  |
| `HF_ACTIVITYNET_FILENAME`                    | Optional filename override inside the ActivityNet snapshot    | ``                      |
| `TASK_DATASET_PATH`                          | Optional local validator dataset path                         | ``                      |
| `TASK_SPLIT`                                 | Validator task split                                          | `validation`            |
| `REQUIRE_ACCESSIBLE_VIDEOS`                  | Skip inaccessible validator task videos                       | `1`                     |
| `TASK_MAX_SAMPLING_ATTEMPTS`                 | Max tries to find an accessible validator task                | `50`                    |
| `VIDEO_AVAILABILITY_CACHE_PATH`              | Legacy base path for validator availability caches            | ``                      |
| `ACCESSIBLE_VIDEO_CACHE_PATH`                | JSON cache path for accessible validator videos               | ``                      |
| `INACCESSIBLE_VIDEO_CACHE_PATH`              | JSON cache path for inaccessible validator videos             | ``                      |
| `VIDEO_AVAILABILITY_CACHE_TTL_HOURS`         | TTL for cached video availability checks                      | `24`                    |
| `VIDEO_AVAILABILITY_TIMEOUT`                 | Timeout for validator-side video availability checks          | `20`                    |
| `ENABLE_SYNTHETIC_EVALUATION`                | Enable synthetic validator scoring and weight updates         | `1`                     |
| `SYNTHETIC_MINER_TIMEOUT_SECONDS`            | Per-miner timeout for synthetic validator evaluation          | `150`                   |
| `MINER_SUBMISSION_CACHE_TTL_SECONDS`         | Cache TTL for loaded v2 submissions                           | `300`                   |
| `MINER_SUBMISSION_REFRESH_INTERVAL_SECONDS`  | Interval between validator submission metadata refreshes      | `60`                    |
| `MINER_SUBMISSION_HEALTH_TIMEOUT_SECONDS`    | Per-runtime timeout for responsive miner `/health` checks     | `10`                    |
| `CHUTES_BASE_DOMAIN`                         | Domain used to resolve `chute_slug` submissions               | `chutes.ai`             |
| `CHRONOSEEK_CHUTES_API_KEY`                  | Optional provider auth token for private Chutes eval          | ``                      |
| `MINER_EMISSION_BURN_PERCENT`                | Percent of miner emissions to burn through UID 0 weighting    | `0`                     |
