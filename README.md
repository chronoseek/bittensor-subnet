# ChronoSeek

> **Google for Videos.**

ChronoSeek is moving from a validator-gateway testnet design to a Chutes-backed `v2.0` `Eval/Serve Split` architecture.

The full design note is [ChronoSeek v2.0 Eval/Serve Split](./docs/CHRONOSEEK_V2_EVAL_SERVE_SPLIT.md).

## Version Guide

### `v1.x` Current deployed testnet design

- miners expose live search services
- validators generate synthetic tasks and score miner responses
- validators may also expose a public gateway for product and developer traffic
- gateway traffic is still forwarded to ranked responsive miners

### `v2.0` Target design: `Eval/Serve Split`

- miners deploy private retrieval runtimes to Chutes
- miners commit structured deployment metadata on-chain
- validators read the latest valid miner submission per hotkey and query the private Chutes directly
- synthetic evaluation remains the only scoring source
- the public API is no longer intended to forward organic requests to miners
- subnet admins can promote selected winning private Chutes into immutable public clones for owner-run API serving
- promoted clones are locked to the exact Docker image that was running at clone time

## Current Scope

### `v1.x` capabilities deployed today

- multimodal baseline with visual retrieval plus transcript-based speech understanding
- deterministic validator evaluation using ActivityNet Captions
- IoU-based scoring with moving-average weight updates
- optional validator gateway for `/search`, `/search/stream`, `/health`, and `/capabilities`

### `v2.0` architecture changes in progress

1. Replace miner-facing organic routing with owner-controlled serving.
2. Treat miner submissions as hosted retrieval runtimes, not only model weights.
3. Resolve miner deployment metadata from chain instead of relying on axon IP and port for search routing.
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

### `v1.x`

```text
User / Client
   │
   ▼
Validator gateway
   ├─ Deterministic evaluation
   └─ Organic / developer query routing
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

### `v1.x` miner runtime

The reference miner currently:

- downloads source video
- performs CLIP-based retrieval
- optionally extracts and transcribes speech
- returns ranked timestamp intervals

### `v2.0` miner submission model

The miner submission unit is shifting from a live axon-backed service toward a Chutes-hosted retrieval runtime with immutable deployment metadata. In practice this means:

- miners still own training and deployment
- miner submissions may include full custom runtime code for video access, clip traversal, transcription, retrieval, ranking, and formatting
- validators still own scoring
- the product API no longer depends on direct miner fanout
- promoted serving uses a Chutes clone locked to the exact Docker image that was running at clone time

## Running Nodes

### Start the miner

```bash
poetry run python miner.py
```

### Start the validator

```bash
poetry run python validator.py
```

These commands remain valid for the current codebase. The routing and deployment semantics described above are the target architecture being implemented next.

## Validator API

### `v1.x` current behavior

Validators can optionally expose:

- `GET /health`
- `GET /capabilities`
- `POST /search`
- `POST /search/stream`

That gateway currently:

- tracks miner liveness from `/health`
- fans out to ranked responsive miners
- aggregates and ranks results
- preserves the shared protocol response shape

### `v2.0` direction

This gateway behavior is being deprecated as the intended public serving model.

The target behavior is:

- validators keep synthetic scoring responsibility
- validators resolve miner Chutes submissions from chain metadata
- owner-run API serves organic traffic from promoted immutable Chutes clones locked to selected Docker images

The validator gateway may still remain useful for testnet and internal diagnostics, but it is no longer the desired product-facing API architecture.

## Environment Variables

The current environment variables still describe the deployed `v1.x` code paths. During the `v2.0` transition, expect new configuration for:

- Chutes submission metadata resolution
- private Chutes authentication
- submission freshness and cache TTLs
- promotion and owner-run serving selection
- active promoted clone selection and rollback

The existing `v1.x` variables remain below until the implementation lands:

| Variable                                     | Description                                                   | Default                 |
| -------------------------------------------- | ------------------------------------------------------------- | ----------------------- |
| `WALLET_NAME`                                | Name of your coldkey                                          | `default`               |
| `HOTKEY_NAME`                                | Name of your hotkey                                           | `default`               |
| `WALLET_PATH`                                | Path to your wallet storage                                   | `~/.bittensor/wallets/` |
| `NETUID`                                     | Subnet NetUID                                                 | `298`                   |
| `NETWORK`                                    | Network (`finney`, `test`, `local`)                           | `test`                  |
| `PORT`                                       | Default value for `axon.port`                                 | `8000`                  |
| `MIN_VALIDATOR_STAKE`                        | Minimum validator stake required by the miner                 | `10000`                 |
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
| `MINER_EMISSION_BURN_PERCENT`                | Percent of miner emissions to burn through UID 0 weighting    | `0`                     |
| `ENABLE_VALIDATOR_API`                       | Enable the optional validator API                             | `0`                     |
| `VALIDATOR_API_HOST`                         | Host for the optional validator API                           | `0.0.0.0`               |
| `VALIDATOR_API_PORT`                         | Port for the optional validator API                           | `8010`                  |
| `VALIDATOR_API_MAX_MINERS`                   | Max miners queried concurrently per validator API request     | `3`                     |
| `VALIDATOR_API_SYNC_MINER_TIMEOUT_SECONDS`   | Per-miner timeout for sync validator API fanout               | `135`                   |
| `VALIDATOR_API_STREAM_MINER_TIMEOUT_SECONDS` | Per-miner timeout for streaming validator API fanout          | `135`                   |
| `VALIDATOR_MINER_HEALTH_INTERVAL_SECONDS`    | Interval between validator liveness sweeps                    | `60`                    |
| `VALIDATOR_MINER_HEALTH_TIMEOUT_SECONDS`     | Per-miner timeout for validator liveness checks               | `5`                     |
