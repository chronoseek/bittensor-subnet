# Validator Anti-Gaming Tasks

## Goal

ChronoSeek validators currently evaluate miners with ActivityNet-derived tasks. The immediate weakness is that a miner can map the public ActivityNet video URL plus raw caption back to public ActivityNet annotations and return the known timestamps without doing video retrieval.

The goal of this implementation is to keep ActivityNet as the validator source of truth while hiding the source dataset identity from miners. Validators should transform each ActivityNet row into a private task artifact before querying miners.

The target flow is:

1. The validator samples ActivityNet internally.
2. The validator selects one caption group and its ground-truth intervals.
3. The validator creates a private cropped clip around the target moment.
4. The validator re-encodes/compresses the clip to strip metadata and reduce size.
5. The validator uploads the task clip to Hippius public S3.
6. The validator sends miners only the Hippius URL, query variant, request ID, protocol version, and `top_k=1`.
7. The validator scores miner responses against clip-local shifted ground truths.

ActivityNet remains the source of truth. Original ActivityNet source URLs, source video IDs, raw captions, and ground-truth timestamps must not be sent to miners or logged at INFO level.

## End-To-End Flow

The validator task flow should run inside the existing validator loop before miner fanout.

1. Load and normalize ActivityNet using the existing dataset loader.
2. Sample a source video/caption with secret-seeded randomness.
3. Check source video accessibility with the existing availability cache.
4. Select a randomized crop window around the target interval group.
5. Download the source video to local validator cache.
6. Crop and re-encode the selected window with ffmpeg.
7. Optionally attempt Vidaio compression if configured.
8. If Vidaio is disabled or fails, use the local ffmpeg-compressed output.
9. Upload the final MP4 to Hippius public S3.
10. Record the upload in the local artifact manifest.
11. Build a protocol-compatible `VideoSearchRequest` using the Hippius URL.
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
| `artifact_manifest.py` | Local JSON manifest for uploaded task clips, expiry, cleanup, and replay metadata. |
| `hardened_task_gen.py` | Orchestrates sampler, query variants, clipper, compressor, storage, and manifest into `generate_task()`. |

Keep provider-specific integration code outside the validator package, parallel to `chronoseek/chain/` and `chronoseek/chutes/`:

| Module | Responsibility |
| --- | --- |
| `chronoseek.hippius.s3` | Hippius S3 upload, download, delete, public URL construction, and credentials validation through the MinIO SDK. |
| `chronoseek.video.vidaio` | Optional Vidaio compression adapter. |

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

Add these validator configuration values to `.env.example` and `validator.py`.

### Task Generation

```env
ENABLE_HARDENED_TASKS=0
VALIDATOR_TASK_SECRET=
VALIDATOR_EVAL_TOP_K=1
TASK_CLIP_CACHE_DIR=~/.cache/chronoseek/task-clips
TASK_ARTIFACT_MANIFEST_PATH=
TASK_ARTIFACT_PREFIX=task-clips
TASK_CLIP_TTL_HOURS=6
TASK_CLIP_CLEANUP_INTERVAL_SECONDS=900
TASK_DELETE_REMOTE_ARTIFACTS=0
TASK_VIDEO_COOLDOWN_HOURS=24
TASK_CAPTION_COOLDOWN_HOURS=168
TASK_QUERY_VARIANTS_PATH=
TASK_MIN_CLIP_DURATION_SECONDS=30
TASK_MAX_CLIP_DURATION_SECONDS=180
TASK_SOURCE_DOWNLOAD_TIMEOUT_SECONDS=120
TASK_ENCODING_PROFILE_NAME=h264-720p-v1
TASK_CLIP_MAX_WIDTH=1280
TASK_CLIP_MAX_HEIGHT=720
TASK_CLIP_VIDEO_BITRATE=1500k
TASK_CLIP_AUDIO_BITRATE=96k
TASK_MAX_PREDICTION_DURATION_SECONDS=60
HARDENED_TASK_MAX_GENERATION_ATTEMPTS=5
```

`ENABLE_HARDENED_TASKS=1` switches the validator from legacy ActivityNet URL tasks to private clipped task artifacts.

`VALIDATOR_TASK_SECRET` is required when hardened task generation is enabled. It should be a long random value and must not be logged.

### Hippius Public S3

```env
HIPPIUS_S3_ENDPOINT_URL=https://s3.hippius.com
HIPPIUS_S3_PUBLIC_BASE_URL=https://s3.hippius.com
HIPPIUS_S3_BUCKET=chronoseek
HIPPIUS_S3_ACCESS_KEY_ID=
HIPPIUS_S3_SECRET_ACCESS_KEY=
HIPPIUS_S3_REGION=decentralized
HIPPIUS_S3_TIMEOUT_SECONDS=60
```

Use Hippius public S3 path-style URLs for miner-facing task clips (`https://s3.hippius.com/chronoseek/task-clips/<video-file-name-or-id>`). Do not use IPFS gateway URLs or presigned URLs for the first implementation.

By default, expired-task cleanup removes local files and manifest entries only. Set `TASK_DELETE_REMOTE_ARTIFACTS=1` only if uploaded Hippius task clips should also be deleted.

### Optional Vidaio

```env
VIDAIO_COMPRESSION_ENABLED=0
VIDAIO_API_BASE_URL=
VIDAIO_API_KEY=
VIDAIO_TIMEOUT_SECONDS=60
```

Vidaio must be optional. If enabled but unavailable, invalid, or timed out, validation must fall back to local ffmpeg compression.

## Step-By-Step Deliverables

### 1. Add Doc And Config Surface

- Add this implementation doc.
- Add validator task, Hippius, scoring hygiene, and optional Vidaio config values to `.env.example`.
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
- Make `run_step` send `top_k=VALIDATOR_EVAL_TOP_K`.

Tests:

- `run_step` accepts legacy tuple tasks.
- `run_step` accepts `ValidationTask`.
- Protocol request uses `ValidationTask.video_url`.
- Protocol request uses `top_k=1` by default.

### 3. Add Secret-Seeded Sampling And Replay Cache

- Implement `task_sampler.py`.
- Seed sampling with `VALIDATOR_TASK_SECRET`, validator hotkey, current block, and nonce.
- Add per-video and per-caption cooldowns.
- Add replay key containing source video, caption, crop bucket, query variant, and encoding profile.

Tests:

- Same secret, block, hotkey, and nonce produce the same candidate.
- Different nonce or secret changes the candidate.
- Replay cache blocks repeated source/caption/crop/query combinations.
- Cooldowns prevent repeated video and caption selection.

### 4. Add Query Variants

- Implement `query_variants.py`.
- Load private variants from `TASK_QUERY_VARIANTS_PATH`.
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

### 6. Add Optional Vidaio Adapter

- Implement `chronoseek.video.vidaio`.
- Define a common `compress(input_path) -> output_path` interface.
- Keep local ffmpeg compression and fallback orchestration in `chronoseek.validator.compression`.
- Vidaio adapter is enabled only by config.
- If Vidaio fails, falls back to local ffmpeg.

Tests:

- Local ffmpeg compressor produces valid MP4.
- Vidaio success path returns compressed output.
- Vidaio timeout/error/invalid output falls back to ffmpeg.
- Validator liveness does not depend on Vidaio.

### 7. Add Hippius Public S3 Upload

- Implement `chronoseek.hippius.s3`.
- Upload final task clips to:

```text
task-clips/<task_id>.mp4
```

- Return public URL using `HIPPIUS_S3_PUBLIC_BASE_URL`.
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

## Rollout Order

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
- Vidaio is optional until a stable API contract is confirmed.
- Short TTL cleanup is the default, with `TASK_CLIP_TTL_HOURS=6`.
- Generated task artifacts and private query variants must not be committed.
- No public protocol change is required.
