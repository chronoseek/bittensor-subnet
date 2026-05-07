# Business Logic & Market Rationale

## 1. Executive Summary

ChronoSeek addresses a persistent problem in video data: users can store and stream video at scale, but they still cannot search inside it reliably by meaning.

The product and subnet are now being separated more cleanly under the `Eval/Serve Split` design:

- the subnet exists to evaluate and rank retrieval runtimes through synthetic tasks
- the product API exists to serve stable organic traffic through owner-controlled infrastructure

This separation improves both decentralization quality and product reliability.

The implementation target is `v2.0`. See [ChronoSeek v2.0 Eval/Serve Split](./CHRONOSEEK_V2_EVAL_SERVE_SPLIT.md) for the source-of-truth architecture.

## 2. Market Opportunity

ChronoSeek still targets the same demand categories:

1. media and entertainment
2. security and surveillance
3. education and e-learning
4. legal and compliance

The user pain remains the same: long-form and unstructured video is still operationally hard to search.

## 3. Competitive Position

ChronoSeek competes differently from centralized incumbents because it can:

- use open competition to improve retrieval quality
- evolve retrieval systems faster than a centrally frozen stack
- separate network incentives from customer-facing serving

That third point is now important. A subnet can be decentralized without forcing customers to depend directly on unstable miner uptime.

## 4. Business Logic By Version

### 4.1. Historical `v1.x` testnet model

- miners provided live retrieval services
- validators ranked miners with synthetic tasks
- validators could also relay organic traffic

This was acceptable for early testing, but weak for a production developer API.

### 4.2. `v2.0` target model: `Eval/Serve Split`

#### Subnet role

- miners deploy private retrieval runtimes to Chutes
- validators evaluate those runtimes with synthetic tasks
- weights and emissions continue to reflect evaluation quality

#### Product role

- the owner runs the public API
- organic traffic is served from promoted immutable Chutes clones, not direct miner fanout
- each promoted clone is locked to the exact Docker image that was running at clone time
- auth, credits, billing, rate limiting, uptime, rollback, and customer support remain product-side responsibilities

## 5. Commercialization Strategy

### 5.1. What the subnet sells indirectly

The subnet does not sell uptime or API contracts directly. It surfaces the best retrieval runtimes.

The economic output of the subnet is:

- ranked retrieval quality
- open competition among miner submissions
- a promotion pipeline for production candidates

### 5.2. What the product sells directly

The product layer sells:

1. public developer API access
2. usage-based search access for customers
3. future enterprise integrations and managed deployments

This is a better commercialization boundary than asking customers to trust live miner endpoints.

## 6. Why The New Design Makes More Sense

The old validator-relay design had a mismatch:

- synthetic evaluation wants decentralization and miner diversity
- product serving wants stability, rollback, observability, and ownership

Those are different operational goals.

`Eval/Serve Split` resolves that mismatch:

- synthetic requests stay decentralized
- organic requests become stable and productized

## 7. Chutes In The Business Model

Chutes is not just an inference helper in the new design. It becomes part of the subnet-to-product handoff:

- miners use private Chutes as submission runtimes
- validators score those private runtimes
- subnet admins can promote selected runtimes into immutable public clones
- promoted clones preserve the full selected runtime by locking the exact Docker image at clone time
- the owner-run API can depend on those promoted artifacts

This gives ChronoSeek a practical bridge from open competition to production serving.

## 8. Flywheel

1. Better miner submissions improve synthetic evaluation results.
2. Better promoted runtimes improve customer-facing API quality.
3. Better product quality attracts more usage and ecosystem attention.
4. More usage supports stronger network incentives and better submissions.

## 9. Risks And Mitigations

### Risk: miner uptime is unstable

Mitigation:

- stop using miners as the production serving plane for organic traffic

### Risk: winning submission is not portable

Mitigation:

- treat miner output as a deployable runtime, not just a score
- require Chutes promotion to clone the selected runtime image rather than reconstructing the runtime from loose model weights
- require structured submission metadata and promotion gates

### Risk: product drifts away from subnet incentives

Mitigation:

- promote from evaluated winners or approved challengers
- keep promotion decisions explicit and auditable

## 10. Bottom Line

The business logic is now cleaner:

- the subnet is an evaluation market
- Chutes is the runtime substrate for private submissions and Docker-image-locked promotion
- the owner-run API is the monetized product surface

That is a stronger design than routing product traffic through validator-owned infrastructure.
