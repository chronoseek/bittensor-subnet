# Problem Statement

## The Gap in Video Search

Users often remember *what happens* in a video, but not *when it happens*. 

Existing tools rely on metadata, subtitles, or manual chaptering and cannot reliably answer semantic queries such as:

> "the scene where two generals fight each other with guns"

**ChronoSeek** solves this problem by building "Google for Videos" in a decentralized, competitive environment.

## Key Challenges

1.  **Metadata Limitations:** Titles and tags don't capture the temporal details of long-form content.
2.  **Labor Intensive:** Manual timestamping (e.g., YouTube chapters) is unscalable for massive archives.
3.  **Semantic Gap:** Keyword search fails on conceptual queries (e.g., searching for "joy" vs. finding a smiling face).
4.  **Centralized Control:** Current advanced search is locked behind "walled gardens" (Google, Apple) with privacy and censorship risks.

## The Solution

A decentralized subnet where miners compete to provide the most accurate timestamp intervals for natural language queries, evaluated by a robust, synthetic ground-truth mechanism.

## v2.0 Serving Boundary

ChronoSeek's production serving model is moving to `Eval/Serve Split`:

- the subnet evaluates miner runtimes with synthetic requests
- miners submit private Chutes-hosted retrieval runtimes
- selected runtimes can be promoted into public Chutes clones
- each promoted clone is locked to the exact Docker image that was running at clone time
- the owner-run API serves organic website and developer traffic from promoted clones, not from live miner fanout

This keeps the subnet decentralized while giving the public API a stable serving backend.
