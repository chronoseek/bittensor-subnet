# Validator Anti-Gaming Tasks

## Status

Hardened ActivityNet-derived task generation is implemented as the default validator task path through `DEFAULT_ENABLE_HARDENED_TASKS=True`. This document describes the implemented flow and keeps the original deliverables as an audit checklist for future changes.

## Goal

ChronoSeek validators evaluate miners with ActivityNet-derived tasks. The immediate weakness in the legacy flow is that a miner can map the public ActivityNet video URL plus raw caption back to public ActivityNet annotations and return the known timestamps without doing video retrieval.

The implemented goal is to keep ActivityNet as the validator source of truth while hiding the source dataset identity from miners. Validators transform each ActivityNet row into a private task artifact before querying miners.

The hardened flow is:

1. The validator samples ActivityNet internally.
2. The validator selects one caption group and its ground-truth intervals.
3. The validator creates a private cropped clip around the target moment, adding
   same-video hard negatives when they fit inside the configured duration budget.
4. The validator re-encodes/compresses the clip with a randomized transform
   profile to strip metadata, vary resolution/bitrate, and reduce size.
5. The validator uses the Vidaio-returned compressed URL when Vidaio succeeds, or uploads the locally compressed fallback clip to Hippius public S3.
6. The validator sends miners only the compressed clip URL, query variant, request ID, protocol version, and `top_k=1`.
7. The validator scores miner responses against clip-local shifted ground truths.
8. At a configurable low rate, the validator converts hardened tasks into
   internal canaries for absent, hard-negative, or repeated-moment probes.
9. The validator records report-only miner telemetry for score, latency,
   failures, task family, canaries, transforms, and repeated prediction shapes.
10. The artifact manifest tracks active exposure counts so validators can limit
    reuse of source videos, captions, query variants, and transform profiles.

ActivityNet remains the source of truth. Original ActivityNet source URLs, source video IDs, raw captions, and ground-truth timestamps must not be sent to miners or logged at INFO level.

## End-To-End Flow

The validator task flow should run inside the existing validator loop before miner fanout.

1. Load and normalize ActivityNet using the existing dataset loader.
2. Sample a source video/caption with secret-seeded randomness.
3. Check source video accessibility with the existing availability cache.
4. Select a randomized crop window around the target interval group.
5. Download the source video to local validator cache.
6. Crop and re-encode the selected window with ffmpeg.
7. If enabled, attempt Vidaio compression using a temporary public task artifact URL.
8. If Vidaio succeeds, send miners the compressed video URL returned by Vidaio.
9. If Vidaio is disabled or fails, use local ffmpeg compression and upload that final MP4 to Hippius public S3.
10. Record the miner-facing artifact URL in the local artifact manifest.
11. Build a protocol-compatible `VideoSearchRequest` using the miner-facing compressed clip URL.
12. Query responsive miners with `top_k=1`.
13. Score only against clip-local shifted ground truths.
14. Periodically clean expired local clips and Hippius objects.

The miner-facing request shape does not change:

```json
{
  "protocol_version": "2026-04-10",
  "request_id": "validation-...",
  "video": {
    "url": "https://s3.hippius.com/chronoseek/task-clips/<task-id>.mp4"
  },
  "query": "query variant text",
  "top_k": 1
}
```

## Modular Code Design

### Internal Model

Add `chronoseek/validator/task_models.py`.

Define `ValidationTask` as the internal task object returned by the hardened generator:

```python
@dataclass(frozen=True)
class ValidationTask:
    task_id: str
    request_id: str
    video_url: str
    query: str
    ground_truths: list[tuple[float, float]]
    clip_duration: float
    source_video_id: str
    source_caption_id: str
    crop_start: float
    crop_end: float
    query_variant_id: str
    artifact_url: str
    artifact_key: str
    expires_at: float
```

Also define `TaskArtifact`, `CropPlan`, and `EncodingProfile` if keeping those concepts separate makes implementation cleaner.

### ActivityNet Loading

Keep `ActivityNetTaskGenerator` focused on loading and normalizing ActivityNet data. It should remain responsible for:

- Hugging Face snapshot loading.
- Local manifest loading.
- Dataset format normalization.
- Grouping repeated captions and multiple timestamp intervals.
- Video accessibility support.

Do not add Hippius, clipping, compression, replay, or query-variant logic to `ActivityNetTaskGenerator`.

### Validator Modules And Integration Boundaries

Keep task-domain logic under `chronoseek/validator/`:

| Module | Responsibility |
| --- | --- |
| `task_models.py` | Dataclasses and internal task/artifact types. |
| `task_sampler.py` | Secret-seeded ActivityNet sampling, cooldowns, replay keys, and diversity checks. |
| `query_variants.py` | Private query variant manifest loading and deterministic fallback variants. |
| `clipper.py` | Source video download, crop planning, ffmpeg crop, metadata stripping, and local re-encode. |
| `compression.py` | Local ffmpeg compression, shared compression result types, and fallback orchestration. |
| `task_transforms.py` | Stable per-task encoding variants, hard-negative metadata, and transform application. |
| `canary_tasks.py` | Canary task selection and mutation policy. |
| `task_exposure.py` | Active-task diversity and exposure limits. |
| `artifact_publication.py` | Provider-neutral publication of local or already-hosted compressed artifacts. |
| `artifact_manifest.py` | Local JSON manifest for miner-facing task artifact URLs, expiry, cleanup, and replay metadata. |
| `task_generator_factory.py` | Constructs legacy or hardened task generators from validator configuration. |
| `hardened_task_gen.py` | Coordinates the hardened task pipeline, retries, cleanup, and manifest recording. |

Keep provider-specific integration code outside the validator package, parallel to `chronoseek/chain/` and `chronoseek/chutes/`:

| Module | Responsibility |
| --- | --- |
| `chronoseek.hippius.s3` | Hippius S3 upload, download, delete, public URL construction, and credentials validation through the MinIO SDK. |
| `chronoseek.video.vidaio_client` | Vidaio HTTP submission, polling, rate-limit handling, and result retrieval. |
| `chronoseek.video.vidaio` | Validator compression adapters for public URLs and temporary Hippius inputs. |

`HardenedActivityNetTaskGenerator` should accept storage and compression interfaces rather than concrete provider classes.

### Compatibility Adapter

Update `BaseTaskGenerator` and `forward.run_step` so validation can consume either:

- Legacy tuple: `(video_url, query, ground_truths)`.
- Hardened object: `ValidationTask`.

The adapter should normalize both into:

```python
NormalizedValidationTask(
    video_url=str,
    query=str,
    ground_truths=list[tuple[float, float]],
    top_k=int,
    clip_duration=float | None,
    task_id=str | None,
)
```

This keeps existing tests and bootstrap/dev flows working while enabling the hardened generator.

## Configuration

The hardened task implementation keeps operator secrets in `.env` and ordinary tuning in `DEFAULT_*` constants in `chronoseek/constants.py`.

### Required Environment

```env
VALIDATOR_TASK_SECRET=
HIPPIUS_S3_ACCESS_KEY_ID=
HIPPIUS_S3_SECRET_ACCESS_KEY=
```

`DEFAULT_ENABLE_HARDENED_TASKS=True` makes the validator use private clipped task artifacts by default. Use `--disable-hardened-tasks` only for bootstrap/debug runs that need legacy ActivityNet URL tasks.

`VALIDATOR_TASK_SECRET` or `--validator-task-secret` and the Hippius credentials are required when hardened task generation is enabled. They should never be logged or committed.

### Code Defaults

Validator defaults such as `DEFAULT_TASK_CLIP_TTL_HOURS`, `DEFAULT_TASK_MIN_CLIP_DURATION_SECONDS`, `DEFAULT_TASK_MAX_CLIP_DURATION_SECONDS`, `DEFAULT_TASK_ENCODING_PROFILE_NAME`, `DEFAULT_VALIDATOR_EVAL_TOP_K`, and the default Hippius endpoint/bucket are constants in `chronoseek/constants.py`.

Use Hippius public S3 path-style URLs for miner-facing task clips (`https://s3.hippius.com/chronoseek/task-clips/<video-file-name-or-id>`). Do not use IPFS gateway URLs or presigned URLs for the first implementation.

By default, expired-task cleanup removes local files and manifest entries only. Use `--task-delete-remote-artifacts` only if uploaded Hippius task clips should also be deleted.

### Vidaio

```env
VIDAIO_COMPRESSION_ENABLED=1
VIDAIO_API_KEY=
VIDAIO_POLL_INTERVAL_SECONDS=15
```

Vidaio compression is enabled by default through the public API. Validators upload a temporary public source clip, submit that URL to Vidaio, send miners the compressed video URL returned by Vidaio, and delete the temporary source object best-effort. The validator does not re-upload successful Vidaio outputs. `VIDAIO_POLL_INTERVAL_SECONDS` defaults to 15 seconds to stay below Vidaio workflow-read rate limits. Set `VIDAIO_COMPRESSION_ENABLED=0` or pass `--disable-vidaio-compression` to use local ffmpeg directly. If Vidaio is disabled, unavailable, invalid, or timed out, validation falls back to local ffmpeg compression and uploads the locally compressed MP4 to Hippius.

## Implementation Checklist

This checklist is retained for implementation review. The current codebase implements these items unless a future note says otherwise.

### 1. Add Doc And Config Surface

- Add this implementation doc.
- Keep validator secrets in `.env.example` and ordinary task/scoring defaults in code constants.
- Add CLI/env parsing in `validator.py`.
- Do not change miner-facing protocol schemas.

Tests:

- Config defaults are parsed correctly.
- Missing optional Vidaio config does not fail validator startup.
- Missing Hippius config fails only when hardened task generation is enabled.

### 2. Add Internal Task Model And Adapter

- Implement `ValidationTask` and related dataclasses.
- Add a normalization helper used by `forward.run_step`.
- Keep legacy tuple task generation working.
- Make `run_step` send `top_k` from the validator configuration.

Tests:

- `run_step` accepts legacy tuple tasks.
- `run_step` accepts `ValidationTask`.
- Protocol request uses `ValidationTask.video_url`.
- Protocol request uses `top_k=1` by default.

### 3. Add Secret-Seeded Sampling And Replay Cache

- Implement `task_sampler.py`.
- Seed sampling with `VALIDATOR_TASK_SECRET` or `--validator-task-secret`, validator hotkey, current block, and nonce.
- Add per-video and per-caption cooldowns.
- Add replay key containing source video, caption, crop bucket, query variant, and encoding profile.

Tests:

- Same secret, block, hotkey, and nonce produce the same candidate.
- Different nonce or secret changes the candidate.
- Replay cache blocks repeated source/caption/crop/query combinations.
- Cooldowns prevent repeated video and caption selection.

### 4. Add Query Variants

- Implement `query_variants.py`.
- Load private variants from the configured query variants path.
- Use deterministic fallback variants only when private variants are unavailable.
- Avoid sending exact ActivityNet captions when a variant exists.
- Log a warning when production uses fallback variants.

Tests:

- Private variant manifest is preferred.
- Exact raw caption is not sent when variants exist.
- Fallback variant selection is deterministic.
- Fallback path logs a warning.

### 5. Add Clipping And Local Compression

- Implement `clipper.py`.
- Download source video using existing video download behavior.
- Select randomized context around ground-truth intervals.
- Crop and re-encode with ffmpeg.
- Strip metadata.
- Bound output resolution and bitrate.
- Shift ground truths to clip-local timestamps.

Tests:

- Crop window includes target interval.
- Shifted ground truths are correct.
- Output MP4 is playable.
- Output metadata does not include source URL or ActivityNet identifiers.
- Local clip cleanup removes temporary source artifacts.

### 6. Add Vidaio Adapter

- Implement `chronoseek.video.vidaio`.
- Define a common `compress(input_path, output_path) -> CompressionResult` interface that can return either a local output path or a hosted compressed URL.
- Keep local ffmpeg compression and fallback orchestration in `chronoseek.validator.compression`.
- Vidaio adapter is enabled by default and can be disabled by config.
- If Vidaio fails, falls back to local ffmpeg.

Tests:

- Local ffmpeg compressor produces valid MP4.
- Vidaio success path returns the hosted compressed video URL.
- Vidaio timeout/error/invalid output falls back to ffmpeg.
- Validator liveness does not depend on Vidaio.

### 7. Add Hippius Public S3 Upload

- Implement `chronoseek.hippius.s3`.
- Upload final task clips to:

```text
task-clips/<task_id>.mp4
```

- Return public URL using the configured Hippius public base URL.
- Validate that bucket, endpoint, access key, secret key, and public base URL are configured.

Tests:

- Upload sends correct object key and content type.
- Public URL is built correctly.
- Upload failure fails task generation cleanly.
- Credentials are not logged.

### 8. Add Artifact Manifest And Cleanup

- Implement `artifact_manifest.py`.
- Track task ID, object key, public URL, created time, expiry time, source task hash, encoding profile, and local paths.
- Cleanup expired local clips and Hippius objects on validator startup.
- Run periodic cleanup in the validator loop.

Tests:

- Manifest records uploaded artifacts.
- Expired artifacts are selected for cleanup.
- Local files are deleted.
- Hippius delete is called for expired objects.
- Cleanup is idempotent.

### 9. Harden Scoring

- Keep continuous IoU as the primary score, but dampen it with interval-shape
  penalties so loose overlapping windows do not score as well as tight
  localizations.
- Apply boundary, center, and duration alignment penalties after selecting the
  best valid top-1 prediction/ground-truth pair.
- Optionally apply a gentle latency multiplier after quality scoring; fast bad
  answers must not outrank accurate answers by speed alone.
- Score only the first valid result.
- Ignore confidence for validator weights.
- Zero invalid responses:
  - non-finite timestamps
  - negative start
  - `end <= start`
  - interval outside clip duration
  - malformed response
  - protocol error
  - timeout
  - empty result
- Add oversized interval protection using `MAX_PREDICTION_DURATION_SECONDS`.
- Ignore duplicate or near-duplicate windows.

Tests:

- Correct clip-local prediction scores by IoU.
- Broad center-aligned intervals score below tighter intervals with comparable
  overlap.
- Center-missed intervals score below center-aligned intervals with comparable
  overlap.
- Original ActivityNet timestamp does not score correctly after clipping.
- Same-video hard-negative candidates are tracked internally and can be included
  in cropped task clips.
- Randomized encoding transform IDs are recorded on internal tasks and included
  in replay identity.
- Absent canaries can reward successful empty responses without rewarding
  timeouts, protocol errors, or ordinary empty responses.
- Miner telemetry summaries and suspicion flags are deterministic and
  report-only until a weight aggregation policy explicitly consumes them.
- Score aggregation exposes quality, reliability, consistency, and suspicion
  components with opt-in post-quality multiplier weights.
- Active task exposure limits can block repeated source/caption/query/transform
  use while previous artifacts remain live.
- Optional latency multipliers preserve zero scores and only dampen otherwise
  valid quality scores.
- Invalid intervals score zero.
- Oversized intervals score zero unless the matching ground-truth span is longer.
- Extra results beyond top-1 do not improve score.

### 10. Harden Logs

- Remove INFO logs for source ActivityNet URL, raw caption, source ID, and ground truths.
- Log task ID, request ID, clip duration, query variant ID, artifact host, ground-truth count, and score summary.
- Keep detailed source metadata only at DEBUG if required for local troubleshooting.

Tests:

- INFO logs do not contain source URL.
- INFO logs do not contain raw caption.
- INFO logs do not contain ground-truth timestamps.
- Score summaries still appear after each validation step.

## Rollout Notes

The rollout order below records the intended staged path. Items 1 through 7 are represented in the current codebase; production enablement and eventual legacy tuple removal remain operator decisions.

1. Ship models, adapters, and tests while keeping legacy tuple generation active.
2. Add scoring hygiene with legacy generator still active.
3. Add query variants and replay cache.
4. Add clipping and local ffmpeg compression.
5. Add Hippius upload behind a disabled hardened-generator flag.
6. Enable hardened generator on local/testnet validators.
7. Add Vidaio optional adapter.
8. Disable original ActivityNet URLs for production.
9. Remove legacy tuple path only after hardened generation is stable.

## Acceptance Criteria

- Validator requests no longer expose original ActivityNet URLs.
- Validator requests no longer send exact ActivityNet captions when variants are available.
- Generated task clips are accessible through Hippius public S3 URLs.
- Scoring uses clip-local timestamps.
- Existing moving-average score aggregation and weight setting remain unchanged.
- Full test suite passes with legacy and hardened task generation paths.

## Explicit Assumptions

- Hippius public S3 URL mode is the default delivery path.
- Local ffmpeg compression is required on validator hosts.
- Vidaio uses the public compression API by default, with local ffmpeg kept as the liveness fallback.
- Short TTL cleanup is the default, with `DEFAULT_TASK_CLIP_TTL_HOURS=6`.
- Generated task artifacts and private query variants must not be committed.
- No public protocol change is required.
