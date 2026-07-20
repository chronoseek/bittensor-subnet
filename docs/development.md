# Owner and Development Guide

This guide is for owner-operated workflows, development checks, and live integration tests. Commands here may call external services, mutate remote state, or consume credits. Run them intentionally.

## Safety Boundary

Use these commands freely for local development:

```bash
poetry run pytest tests/unit
poetry run python scripts/verify_miner_retrieval.py
poetry run python scripts/verify_hardened_task_generation.py --use-smoke-dataset --no-require-accessible-videos
poetry run python scripts/test_chutes_runtime_local.py --print-commands
```

Commands in [Production Chutes Helper](#production-chutes-helper) and [Live Module Tests](#live-module-tests) are intentionally live. They may use Chutes credits, write to Hippius, or commit/fetch chain state.

## Module Boundaries

ChronoSeek keeps provider and domain logic separated:

| Module | Responsibility |
| --- | --- |
| `chronoseek.chain.submissions` | On-chain miner metadata commit/fetch. |
| `chronoseek.chutes.deployment` | Chutes API image build, deploy, metadata extraction, and deployment helper logic. |
| `chronoseek.chutes.runtime` | Chutes endpoint resolution, auth headers, and runtime health checks. |
| `chronoseek.hippius.s3` | Hippius S3-compatible upload, download, delete, public URL construction, and URL parsing through MinIO. |
| `chronoseek.video.downloader` | Shared validator/miner video acquisition routing. |
| `chronoseek.video.vidaio_client` | Vidaio HTTP workflow, polling, rate-limit handling, and result retrieval. |
| `chronoseek.video.vidaio` | Vidaio compressor adapters for URL and storage-backed inputs. |
| `chronoseek.validator.task_generator_factory` | Validator task-generator dependency construction. |
| `chronoseek.validator.task_transforms` | Hardened encoding variants and transform metadata. |
| `chronoseek.validator.artifact_publication` | Provider-neutral local and hosted artifact publication. |
| `chronoseek.validator.*` | Dataset loading, task sampling, canaries, exposure limits, manifests, scoring, and validator orchestration. |

Legacy compatibility modules for video downloader, validator storage, and Vidaio were removed. Use the canonical modules above.

## Unit Tests

Standard unit tests live under `tests/unit/`. Default pytest discovery is configured to use that directory, so live integration checks are not collected unless called explicitly.

```bash
poetry run pytest tests/unit
```

## Local Verification Scripts

Inspect hardened validator task generation without querying miners:

```bash
poetry run python scripts/verify_hardened_task_generation.py --use-smoke-dataset --no-require-accessible-videos
```

Use real ActivityNet/Hippius configuration by omitting `--use-smoke-dataset` and adding `--upload-to-hippius` when you want to test the public S3 upload path. The script prints the original source task, generated `ValidationTask`, miner-facing request, and local artifact paths.

Run the local reference miner retrieval pipeline against a sampled or custom task:

```bash
poetry run python scripts/verify_miner_retrieval.py
```

## Local Chutes Runtime Testing

Local Chutes checks avoid production Chutes API calls and should be used before production build/deploy.

The local helper defaults to `chronoseek_chute_local:chute`, which installs the current working tree into the container so local runtime edits are exercised. Production deploys should use `chronoseek_chute:chute`, copied from `chronoseek_chute.example.py` and edited for the immutable package or revision you intend to publish.

Print local build/run commands:

```bash
poetry run python scripts/test_chutes_runtime_local.py --print-commands
```

Build locally:

```bash
poetry run python scripts/test_chutes_runtime_local.py --build
```

Run the local Chutes container in a separate terminal:

```bash
poetry run python scripts/test_chutes_runtime_local.py --run
```

Smoke-test the running local runtime:

```bash
poetry run python scripts/test_chutes_runtime_local.py --smoke
```

Health-only smoke test:

```bash
poetry run python scripts/test_chutes_runtime_local.py --smoke --health-only
```

The helper follows the Chutes local flow: `chutes build <ref> --local`, then `docker run ... chutes run <ref> --dev`.

## Production Chutes Helper

Production build/deploy commands call Chutes APIs and may consume credits.

Build and deploy:

```bash
poetry run python scripts/deploy_chutes_runtime.py --build --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee \
  --artifact-id chronoseek-runtime
```

Deploy only after the image exists:

```bash
poetry run python scripts/deploy_chutes_runtime.py --deploy \
  --chute-ref chronoseek_chute:chute \
  --accept-fee
```

Print the suggested on-chain commit command without build/deploy:

```bash
poetry run python scripts/deploy_chutes_runtime.py \
  --chute-ref chronoseek_chute:chute
```

Lookup existing Chutes metadata:

```bash
poetry run python scripts/deploy_chutes_runtime.py --lookup-only \
  --chute-slug alice-chronoseek-runtime-<timestamp>
```

If an image already exists, the helper prompts before deleting and rebuilding. Use `--overwrite-existing-image` for non-interactive rebuilds or `--no-overwrite-existing-image` to fail fast.

## Live Module Tests

Live tests are under `tests/live/` and are skipped unless `CHRONOSEEK_LIVE_TESTS=1` plus a module-specific flag are set. Run one module at a time.

The files are split by external module:

| File | Scope |
| --- | --- |
| `tests/live/test_video_download.py` | Shared video downloader. |
| `tests/live/test_hippius_storage.py` | Hippius S3 upload/download. |
| `tests/live/test_chutes.py` | Chutes build/deploy. |
| `tests/live/test_chain.py` | Chain commit/fetch. |
| `tests/live/test_vidaio.py` | Vidaio compression. |

### Video Download

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_VIDEO_DOWNLOAD=1 \
LIVE_VIDEO_DOWNLOAD_URL="https://www.youtube.com/watch?v=<id>" \
poetry run pytest tests/live/test_video_download.py::test_live_video_download -s
```

### Hippius Upload/Download

This check is split into two explicit steps: ensure the bucket exists and has the public-read policy, upload with validator credentials, then download the public Hippius URL through the miner downloader path without Hippius credentials. Uploaded objects are kept for later inspection.

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_HIPPIUS=1 \
LIVE_HIPPIUS_SOURCE_VIDEO_URL="https://www.youtube.com/watch?v=<id>" \
LIVE_HIPPIUS_OBJECT_KEY="task-clips/live-test-manual.mp4" \
HIPPIUS_S3_ACCESS_KEY_ID="<access-key>" \
HIPPIUS_S3_SECRET_ACCESS_KEY="<secret-key>" \
HIPPIUS_S3_BUCKET="chronoseek" \
poetry run pytest tests/live/test_hippius_storage.py::test_01_live_hippius_upload -s
```

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_HIPPIUS=1 \
LIVE_HIPPIUS_OBJECT_KEY="task-clips/live-test-manual.mp4" \
HIPPIUS_S3_BUCKET="chronoseek" \
poetry run pytest tests/live/test_hippius_storage.py::test_02_live_hippius_download -s
```

### Chutes Build

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_CHUTES=1 \
CHRONOSEEK_LIVE_CHUTES_BUILD=1 \
CHUTES_API_KEY="<chutes-api-key>" \
LIVE_CHUTES_CHUTE_REF="chronoseek_chute:chute" \
LIVE_CHUTES_CHUTE_SLUG="alice-chronoseek-runtime-live" \
poetry run pytest tests/live/test_chutes.py::test_live_chutes_build -s
```

### Chutes Deploy

Deploy requires an extra fee confirmation flag:

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_CHUTES=1 \
CHRONOSEEK_LIVE_CHUTES_DEPLOY=1 \
CHRONOSEEK_LIVE_CHUTES_ACCEPT_FEE=1 \
CHUTES_API_KEY="<chutes-api-key>" \
LIVE_CHUTES_CHUTE_REF="chronoseek_chute:chute" \
LIVE_CHUTES_CHUTE_SLUG="alice-chronoseek-runtime-live" \
poetry run pytest tests/live/test_chutes.py::test_live_chutes_deploy -s
```

### Chain Commit

This check requires a miner hotkey with no revealed ChronoSeek runtime submission yet. Miner submissions are permanent per hotkey.

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_CHAIN=1 \
CHRONOSEEK_LIVE_CHAIN_COMMIT=1 \
WALLET_NAME="<coldkey>" \
HOTKEY_NAME="<hotkey>" \
NETWORK="finney" \
NETUID="<netuid>" \
LIVE_CHAIN_CHUTE_SLUG="alice-chronoseek-runtime-live" \
poetry run pytest tests/live/test_chain.py::test_live_chain_commit_submission -s
```

### Duplicate Chain Commit Rejection

This check requires a miner hotkey that already has a revealed ChronoSeek runtime submission. It verifies the local commit path rejects a second submission before sending another chain commitment. If a duplicate commitment does appear on-chain for a hotkey, validators disqualify that hotkey and force its score to zero.

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_CHAIN=1 \
CHRONOSEEK_LIVE_CHAIN_DUPLICATE_COMMIT=1 \
WALLET_NAME="<coldkey>" \
HOTKEY_NAME="<already-submitted-hotkey>" \
NETWORK="finney" \
NETUID="<netuid>" \
poetry run pytest tests/live/test_chain.py::test_live_chain_rejects_duplicate_submission_for_hotkey -s
```

### Chain Fetch

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_CHAIN=1 \
NETWORK="finney" \
NETUID="<netuid>" \
LIVE_CHAIN_EXPECTED_HOTKEY="<ss58-hotkey>" \
LIVE_CHAIN_EXPECTED_DUPLICATE_HOTKEY="<optional-duplicate-ss58-hotkey>" \
poetry run pytest tests/live/test_chain.py::test_live_chain_fetch_submissions -s
```

### Vidaio

```bash
CHRONOSEEK_LIVE_TESTS=1 \
CHRONOSEEK_LIVE_VIDAIO=1 \
VIDAIO_API_KEY="<vidaio-api-key>" \
VIDAIO_POLL_INTERVAL_SECONDS=15 \
LIVE_VIDAIO_INPUT_URL="https://example.com/input.mp4" \
poetry run pytest tests/live/test_vidaio.py::test_live_vidaio_compress -s
```

## Live Test Flags

| Flag | Scope |
| --- | --- |
| `CHRONOSEEK_LIVE_TESTS=1` | Enables live tests globally. |
| `CHRONOSEEK_LIVE_VIDEO_DOWNLOAD=1` | Video downloader live check. |
| `CHRONOSEEK_LIVE_HIPPIUS=1` | Hippius upload/download live check. |
| `CHRONOSEEK_LIVE_CHUTES=1` | Chutes live checks. |
| `CHRONOSEEK_LIVE_CHUTES_BUILD=1` | Chutes build check. |
| `CHRONOSEEK_LIVE_CHUTES_DEPLOY=1` | Chutes deploy check. |
| `CHRONOSEEK_LIVE_CHUTES_ACCEPT_FEE=1` | Confirms Chutes deploy fee acceptance. |
| `CHRONOSEEK_LIVE_CHAIN=1` | Chain live checks. |
| `CHRONOSEEK_LIVE_CHAIN_COMMIT=1` | Chain commit check. |
| `CHRONOSEEK_LIVE_CHAIN_DUPLICATE_COMMIT=1` | Duplicate miner submission rejection check. |
| `LIVE_CHAIN_EXPECTED_DUPLICATE_HOTKEY=<ss58-hotkey>` | Optional chain fetch assertion that a known duplicate-submitted hotkey is disqualified and absent from valid submissions. |
| `CHRONOSEEK_LIVE_VIDAIO=1` | Vidaio compression live check. |
