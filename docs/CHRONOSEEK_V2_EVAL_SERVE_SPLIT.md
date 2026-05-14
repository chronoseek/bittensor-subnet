# ChronoSeek v2.0 Eval/Serve Split

## Status

`v2.0` is the active next architecture track for ChronoSeek.

The confirmed Chutes behavior is:

- a subnet admin can clone a selected miner Chute for public serving
- the clone is locked to the exact Docker image that was running at clone time
- the promoted clone is owner-operated and no longer depends on the miner staying online

This makes it feasible to promote a full ChronoSeek retrieval runtime, not only model weights.

## One-Line Product Explainer

ChronoSeek is Google for Videos.

## Mechanism Name

Use the engineering name:

- `Eval/Serve Split`

Meaning:

- `eval plane`: decentralized subnet evaluation through synthetic validator tasks
- `serve plane`: owner-controlled public API serving organic user and developer traffic

## Versioning

### `v1.x`

Earlier testnet behavior:

- miners ran live search services
- validators generated synthetic tasks and queried miner endpoints
- validators scored responses and set weights
- validators could optionally relay organic requests to responsive miners

This was useful for testnet, but it is not the production API model.

### `v2.0`

Target implementation:

- miners deploy full private retrieval runtimes to Chutes
- miners commit structured Chutes submission metadata on-chain
- validators resolve the latest valid submission per hotkey from chain state
- validators query private miner Chutes for synthetic evaluation only
- the public API is owner-run and serves organic traffic from promoted immutable Chutes clones
- miners and validators do not publish ports through subnet metadata
- promoted clones are locked to the exact Docker image that was running when cloned

## Core Design

The subnet should evaluate what miners build. The public API should serve customers from infrastructure the owner controls.

```text
Synthetic evaluation path:
Validator
  -> latest on-chain miner submission
  -> private miner Chute
  -> synthetic score
  -> on-chain weights

Organic serving path:
User / Developer
  -> owner-run API
  -> promoted immutable Chutes clone
  -> stable product response
```

## Miner Submission Unit

The v2.0 submission unit is a full retrieval runtime, not just a model.

A miner runtime may include:

- video download and access handling
- clip traversal
- frame or embedding extraction
- transcription
- retrieval and ranking logic
- response formatting

Because Chutes promotion locks the exact Docker image, the full custom runtime can be promoted as the public-serving artifact.

## Recommended On-Chain Manifest

Validators should not rely on raw IPs or mutable miner-hosted URLs for v2 evaluation. Miners should commit a structured manifest.

Recommended minimal manifest:

```json
{
  "version": "2.0",
  "runtime": "chutes",
  "protocol": "chronoseek-runtime-v2",
  "hotkey": "miner-hotkey-ss58",
  "chute_id": "chute-deployment-id",
  "chute_slug": "chronoseek-chronoseek-runtime-20260510143015999",
  "created_at_block": 123456
}
```

Optional fields:

```json
{
  "model_repo": "owner/repo",
  "model_revision": "immutable-revision",
  "artifact_digest": "sha256:...",
  "capabilities": ["video-url-search", "top-k-results"]
}
```

The implementation should prefer a canonical Chutes identifier such as `chute_id`. Validators can then resolve the serving slug or endpoint from Chutes metadata.

Current subnet implementation uses latest revealed chain commitments by miner hotkey. There is no local metadata-file fallback, including for testing.

For now, validators resolve an explicit `endpoint` first, then `chute_slug` as `https://{chute_slug}.${CHUTES_BASE_DOMAIN}`. Responsive miner selection requires both valid revealed metadata for a registered metagraph hotkey and a successful `/health` response from the resolved runtime. A `chute_id` remains part of the canonical identity, but a validator needs either a direct endpoint, a slug, or future Chutes metadata lookup to dispatch a request.

Chutes derives the public slug from the Chutes account and API `name`, so the deploy helper appends a UTC millisecond timestamp to the API `name` while preserving brand casing, for example `ChronoSeek-runtime-20260510143015999`. The committed `chute_slug` is unique and includes the lowercase Chutes account plus timestamp, for example `chronoseek-chronoseek-runtime-20260510143015999`. The description/tagline is `ChronoSeek`; the human display label in logs/readme is `ChronoSeek Runtime`; the visible author comes from the Chutes account. Chutes deployment naming is off-chain and the deploy helper prints the slug to commit with `miner.py`. The deploy helper also uploads the ChronoSeek logo from `https://chronoseek.org/logo.png` and reuses the returned Chutes `logo_id` for the image and chute payloads. For miner-side video access, the image build step copies `YTDLP_COOKIES` into `/opt/chronoseek/miner-files/ytdlp/` when the local file exists, rewrites the cookie env var inside the image to the container path, installs Deno for yt-dlp EJS challenge solving, and forwards `HF_TOKEN` when it is present in the build environment.

Local validation should use `scripts/test_chutes_runtime_local.py`, which follows the Chutes local testing flow and avoids production API calls or credit usage. `scripts/deploy_chutes_runtime.py` is for intentional production Chutes API build/deploy only.

Miners can commit a submission with:

```bash
poetry run python miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --chute-id chute-deployment-id \
  --chute-slug chronoseek-chronoseek-runtime-20260510143015999 \
  --artifact-revision immutable-revision
```

## Validator Responsibilities

In v2.0, validators should:

- generate synthetic tasks as before
- read the metagraph and latest revealed miner commitments
- parse and validate each miner's Chutes manifest
- resolve the Chutes endpoint for each valid hotkey
- query miner Chutes only for synthetic evaluation
- score outputs with the existing scoring loop initially
- set weights as before

Validators should not:

- forward organic user or developer traffic as the intended production path
- treat miner uptime as a product SLA
- make organic traffic part of rewards or weights

## Public API Responsibilities

The public API should:

- authenticate website users and developer API keys
- enforce credits, rate limits, billing, and usage records
- route organic traffic only to owner-controlled promoted Chutes clones
- support promotion, rollback, and champion selection
- keep customer-facing uptime independent of miner churn

The public API should not:

- forward organic traffic to live miner endpoints
- reveal private miner Chute endpoints
- require customers to understand subnet routing

## Promotion Flow

```text
1. Miner deploys private Chute.
2. Miner commits Chutes metadata on-chain.
3. Validators evaluate the private Chute synthetically.
4. Owner selects a winner, champion, or approved candidate.
5. Chutes clones the selected private Chute.
6. The clone is locked to the exact Docker image running at clone time.
7. Owner API routes organic traffic to the promoted clone.
8. Owner can roll back to a previous promoted clone if needed.
```

## Implementation Order

1. Define the v2 manifest parser and validator. `Implemented`
2. Add chain read logic for latest revealed commitments by hotkey. `Implemented`
3. Resolve direct endpoints and `chute_slug` values to Chutes endpoints. `Implemented`
4. Resolve `chute_id` through Chutes metadata when provider lookup is available. `Pending`
5. Add synthetic validator routing to Chutes endpoints. `Implemented`
6. Keep scoring and task generation unchanged. `Implemented`
7. Remove local validator serving and local miner serving paths. `Implemented`
8. Add platform configuration for promoted serving backend selection.
9. Add owner-admin promotion records and rollback selection.

## Non-Goals For First v2.0 Cut

- changing the reward function
- changing the public search request/response protocol
- making organic traffic affect scores
- building a marketplace of public deployments
- requiring miners to publish stable public APIs themselves

## Design Rationale

The old model mixed two different goals:

- subnet evaluation needs decentralized miner competition
- public serving needs stability, observability, rollback, and ownership

`Eval/Serve Split` keeps the incentive market decentralized while making the product API operationally stable.
