# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ChronoSeek is a Bittensor subnet for semantic video moment retrieval ("Google for Videos"): given a video and a natural-language query, miners return the best-matching timestamp interval. It uses the `v2.0` Decoupled Evaluation-Serving Architecture (DESA):

- **Miners** deploy private retrieval runtimes on Chutes and commit structured runtime metadata on-chain (permanent per hotkey — a hotkey with multiple revealed submissions is disqualified).
- **Validators** (`validator.py`) read those on-chain submissions, generate synthetic video-search tasks, query miner Chutes for evaluation, score responses, and set weights.
- **Organic serving is separate from subnet scoring** — owner-run promoted Chutes runtimes serve real user/developer traffic, not validator fanout.

Full design notes: [docs/CHRONOSEEK_V2_DESA.md](docs/CHRONOSEEK_V2_DESA.md), [docs/DESIGN.md](docs/DESIGN.md), [docs/BUSINESS_LOGIC.md](docs/BUSINESS_LOGIC.md). Role guides: [docs/validator.md](docs/validator.md), [docs/miner.md](docs/miner.md), [docs/development.md](docs/development.md).

## Commands

Install:
```bash
poetry install
cp .env.example .env   # then edit for your role
```

Run:
```bash
poetry run python validator.py
poetry run python miner.py --wallet.name <name> --wallet.hotkey <hotkey> --chute-slug <slug>
```

Unit tests (default pytest discovery is `tests/unit` only):
```bash
poetry run pytest tests/unit
poetry run pytest tests/unit/test_scoring.py
poetry run pytest tests/unit/test_scoring.py::test_name  # single test
```

Local, non-mutating verification scripts — safe to run freely:
```bash
poetry run python scripts/verify_miner_retrieval.py
poetry run python scripts/verify_hardened_task_generation.py --use-smoke-dataset --no-require-accessible-videos
poetry run python scripts/test_chutes_runtime_local.py --print-commands   # also --build / --run / --smoke
```

Live/mutating checks (`tests/live/`, real Chutes/Hippius/chain calls, may cost credits) require explicit opt-in env flags (`CHRONOSEEK_LIVE_TESTS=1` plus a module flag) and are documented in [docs/development.md](docs/development.md) — don't run these without deliberate intent.

Production Chutes deploy (calls the live Chutes API):
```bash
poetry run python scripts/deploy_chutes_runtime.py --build --deploy --chute-ref chronoseek_chute:chute --accept-fee --artifact-id chronoseek-runtime
```

## Architecture

### Module boundaries (`chronoseek/`)

| Module | Responsibility |
| --- | --- |
| `chain.submissions` | On-chain miner runtime-metadata commit/fetch. |
| `chutes.deployment` | Chutes image build, deploy, metadata extraction. |
| `chutes.runtime` | Chutes endpoint resolution, auth headers, health checks. |
| `hippius.s3` | Hippius S3-compatible storage (upload/download/delete/URL parsing via MinIO). |
| `video.downloader` | Shared validator/miner video acquisition, routed by URL type (Hippius / yt-dlp / raw HTTP), with strategy fallback. |
| `video.vidaio_client` / `video.vidaio` | Vidaio compression HTTP workflow and adapters; falls back to local ffmpeg if Vidaio fails. |
| `epistula.py` | Request signing/verification (Epistula headers) — the HTTP transport auth boundary between validators and miner Chutes. |
| `protocol_models.py` | Pydantic request/response schemas, versioned via `PROTOCOL_VERSION`. |
| `validator.*` | Task generation (`task_gen`, `hardened_task_gen`, `base_task_gen`), sampling, canaries, exposure limits, artifact manifest/publication, aggregation, scoring, telemetry, orchestration (`forward.py`). |
| `miner.*` | Runtime auth, request logic (`logic.py`), Chutes runtime entrypoint (`runtime.py`), audio/frame/transcript/clip extraction utilities. |

Legacy compatibility modules for video downloader, validator storage, and Vidaio were removed — always use the canonical modules above, not older equivalents that may be referenced in old docs/history.

### Key flows

- **Validator loop**: `validator.py` orchestrates via `chronoseek/validator/forward.py` — sample/generate a task (hardened tasks are on by default and require `VALIDATOR_TASK_SECRET` + Hippius credentials), query miner Chutes over Epistula-signed HTTP, score with `scoring.py`, aggregate, set weights.
- **Miner runtime**: deployed as a Chutes container built from `chronoseek_chute.py` (copy from `chronoseek_chute.example.py`); `miner.py` is the CLI that commits the deployed Chutes slug on-chain. The runtime itself lives in `chronoseek/miner/`.
- **Video access**: both validator task generation and miner runtimes go through `chronoseek/video/downloader.py`, which tries Hippius → yt-dlp → raw HTTP in order and returns a structured error if all fail.
- Runtime tuning defaults are centralized in `chronoseek/constants.py`; `.env` is reserved for identity, credentials, secrets, and host-specific file paths (see `.env.example`), not tuning knobs. CLI flags can override constants.
