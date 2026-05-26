# ChronoSeek

ChronoSeek is **Google for Videos**: a Bittensor subnet for semantic video moment retrieval.

Given a video and a natural-language query, ChronoSeek miners return the timestamp interval that best matches the query. Validators generate synthetic video-search tasks, score miner responses, and set on-chain weights.

## Architecture

ChronoSeek uses the `v2.0` Decoupled Evaluation-Serving Architecture (DESA):

- **Miners** deploy private retrieval runtimes on Chutes.
- **Miners** commit structured runtime metadata on-chain.
- **Validators** read those submissions, query miner Chutes for synthetic evaluation, score responses, and set weights.
- **Owner-run serving** is separate from subnet scoring. Organic user and developer traffic is served by operator-controlled promoted Chutes runtimes, not by validator fanout.

The full design note is [ChronoSeek v2.0 DESA](./docs/CHRONOSEEK_V2_DESA.md).

## Guides

Start with the guide for your role:

- [Validator guide](./docs/validator.md): validator setup, required environment variables, hardened task generation, Hippius storage, and Vidaio compression.
- [Miner guide](./docs/miner.md): Chutes runtime deployment, video download behavior, and on-chain runtime metadata commits.
- [Owner and development guide](./docs/development.md): local Chutes testing, production build/deploy helpers, live integration checks, and module boundaries.

Additional background:

- [Problem statement](./docs/PROBLEM_STATEMENT.md)
- [System design](./docs/DESIGN.md)
- [Business logic](./docs/BUSINESS_LOGIC.md)
- [Validator anti-gaming tasks](./docs/VALIDATOR_ANTI_GAMING_TASKS.md)

## Install

Prerequisites:

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)

Clone and install:

```bash
git clone https://github.com/chronoseek/bittensor-subnet.git
cd bittensor-subnet
poetry install
```

Create your environment file:

```bash
cp .env.example .env
```

Then edit `.env` for your role. The detailed required variables live in the validator and miner guides.

## Quick Start

Run a validator:

```bash
poetry run python validator.py
```

Run the miner metadata commit after deploying a Chutes runtime:

```bash
poetry run python miner.py \
  --wallet.name <wallet-name> \
  --wallet.hotkey <hotkey-name> \
  --chute-slug alice-chronoseek-runtime-<timestamp>
```

The examples assume Chutes account `alice` and Hippius bucket `chronoseek`.

## Video Access

Both validator-side task generation and miner runtimes need to fetch videos. The shared downloader routes by URL type:

- Hippius URLs use the Hippius S3-compatible API.
- YouTube, Vimeo, and other extractor-platform URLs use `yt-dlp`.
- Direct media URLs use raw HTTP.
- If a strategy fails, the runtime falls back to the next supported strategy and returns a structured error when all strategies fail.

Validator hardened tasks upload generated clips to public Hippius URLs shaped like:

```text
https://s3.hippius.com/chronoseek/task-clips/<video-file-name-or-id>
```

## Current Scope

The active subnet path is synthetic evaluation:

```text
Validator
  -> chain submission metadata
  -> private miner Chutes
  -> scoring
  -> on-chain weights
```

Organic serving is intentionally outside validator scoring:

```text
User / Developer
  -> owner-run API
  -> promoted immutable Chutes runtime
  -> stable product response
```
