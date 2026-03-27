# ChronoSeek

> **A Bittensor subnet for decentralized semantic video moment retrieval.**

**ChronoSeek** enables semantic search over video content by mapping natural-language scene descriptions to precise timestamp intervals within a video.

---

## ⚠️ MVP Disclaimer (Hackathon)

**Current Status:** Proof of Concept (MVP)

This repository represents the initial **Minimum Viable Product (MVP)** implementation of the ChronoSeek protocol.

**MVP Scope Limitations:**

- **Model:** Miners currently use a baseline **CLIP (ViT-B/32)** sliding window approach. This is computationally expensive and not optimized for long-form video.
- **Dataset:** Validators evaluate against **ActivityNet Captions** annotations. For local verification and smoke tests, the repo also includes a small curated fixture with directly downloadable sample videos.
- **Scoring:** Validators score miners by best-match Intersection-over-Union (IoU) in `[0, 1]` and maintain moving averages for weight setting. A strict IoU threshold of `0.5` is still used in local verification scripts when you want pass/fail semantics.
- **Inference:** All inference happens locally on the miner.

**Future Enhancements (Roadmap):**

1.  **Modular Inference:** Integration with **Chutes (SN64)** for serverless, verifiable model execution.
2.  **SOTA Models:** Transition to temporal-aware architectures like **Moment-DETR** or **VideoLlama**.
3.  **Synthetic Tasks:** Implementation of a VLM-based Oracle (using GPT-4o or Gemini) to generate infinite synthetic training tasks from any video URL.
4.  **Vector Caching:** Miners will implement vector databases (Milvus/Chroma) to cache video embeddings, enabling millisecond-level retrieval for repeated queries.

---

## 📚 Project Documentation

This project is organized into the following key documents:

- **[Problem Statement](docs/PROBLEM_STATEMENT.md)**  
  _Why this subnet exists, the "dark data" problem, and the limitations of current search tools._

- **[System Design](docs/DESIGN.md)**  
  _Technical architecture, including Miner/Validator logic, Synthetic Task Generation, and SOTA research references (CLIP, LLMs)._

- **[Business Logic & Market Rationale](docs/BUSINESS_LOGIC.md)**  
  _Market size ($94B+), commercialization strategy, and competitive advantage against centralized giants._

---

## 🚀 Quick Start (Hackathon)

### 1. The Core Concept

We are building a decentralized protocol where:

- **Miners** use AI models (CLIP, Transformers) to "watch" videos and find specific moments.
- **Validators** generate synthetic queries to grade miners and serve organic requests.
- **Users** get precise timestamps (e.g., "04:12 - 04:18") for natural language queries.

### 2. Architecture Overview

```
User / Client
   │
   ▼
Validator (Gateway)
   ├─ Synthetic evaluation (scoring & weights)
   └─ Organic query routing
   │
   ▼
Miners
   └─ Semantic video analysis (CLIP / SOTA Models)
```

## 📦 Installation & Setup

This project uses `poetry` for dependency management.

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) installed

### 1. Clone & Install

```bash
git clone https://github.com/chronoseek/bittensor-subnet.git
cd bittensor-subnet

# Install dependencies and create virtualenv
poetry install
```

### 2. Activate Virtual Environment

```bash
poetry env activate
```

### 3. Set up HuggingFace Token

To download models (e.g., CLIP), you need a Hugging Face token.

1. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Set it in your environment:

```bash
export HF_TOKEN=your_token_here
```

_Or add it to your `.env` file._

## 🏃‍♂️ Running your nodes (Testnet: SN298, Mainnet: TBD)

### 1. Start the Miner

The miner listens for HTTP requests from validators.

```bash
# Starts miner on port 8000
poetry run python miner.py
```

_Ensure your wallet/hotkey is registered on SN298._

### 2. Start the Validator

The validator generates synthetic tasks, queries miners, and scores them.

```bash
# Starts validator loop
poetry run python validator.py
```

_Ensure your wallet/hotkey is registered on SN298._

### 2a. Optional Validator API

Validators can optionally expose a public API for application or developer use. This is disabled by default.

Supported endpoints:

- `GET /health`
- `POST /search`

The `/search` endpoint accepts the standard ChronoSeek `VideoSearchRequest` payload and returns a standard `VideoSearchResponse`. When gateway-level failures occur, the validator returns structured protocol errors using the same `ProtocolError` envelope.

Gateway behavior:

- the validator queries several miners, ranked by the validator's current moving scores
- it aggregates the returned windows across those miners
- it returns the top `k` ranked windows by confidence in the standard `VideoSearchResponse.results` field
- the response remains compatible with the shared protocol contract in the `git/protocol` repo

Example:

```bash
poetry run python validator.py \
  --enable-validator-api \
  --validator-api-host 0.0.0.0 \
  --validator-api-port 8010
```

### 3. Local Miner Search Test

You can test `miner.py` directly without running a validator by sending a signed
Epistula request to `/search` with `scripts/test_miner_search.py`.

```bash
# In terminal A: start miner
poetry run python miner.py

# In terminal B: run a signed search request
poetry run python scripts/test_miner_search.py \
  --video-url "https://www.w3schools.com/html/mov_bbb.mp4" \
  --query "people talking"
```

Optional flags:

- `--endpoint` (default: `http://127.0.0.1:8000/search`)
- `--top-k` (default: `3`)
- `--wallet-name`, `--wallet-hotkey`, `--wallet-path` (for Epistula signing key)

## ⚙️ Running with PM2 (Production)

For long-running processes, use [PM2](https://pm2.keymetrics.io/).

### 1. Install PM2

```bash
npm install pm2 -g
```

### 2. Start Miner

```bash
pm2 start "poetry run python miner.py --wallet.name default --wallet.hotkey default" --name miner
```

### 3. Start Validator

```bash
pm2 start "poetry run python validator.py --wallet.name default --wallet.hotkey default" --name validator
```

### 4. Manage Processes

```bash
pm2 list
pm2 logs miner
pm2 logs validator
```

## 🔧 Environment Variables

| Variable                             | Description                                                    | Default                 |
| ------------------------------------ | -------------------------------------------------------------- | ----------------------- |
| `WALLET_NAME`                        | Name of your coldkey                                           | `default`               |
| `HOTKEY_NAME`                        | Name of your hotkey                                            | `default`               |
| `WALLET_PATH`                        | Path to your wallet storage                                    | `~/.bittensor/wallets/` |
| `NETUID`                             | Subnet NetUID                                                  | `298` (Mainnet TBD)     |
| `NETWORK`                            | Network (finney, test, local)                                  | `test`                  |
| `PORT`                               | Default value for `axon.port`                                  | `8000`                  |
| `MIN_VALIDATOR_STAKE`                | Minimum validator stake required by the miner                  | `10000`                 |
| `LOG_LEVEL`                          | Logging verbosity                                              | `INFO`                  |
| `HF_TOKEN`                           | Hugging Face Token                                             | `None`                  |
| `HF_HOME`                            | Hugging Face cache directory                                   | `~/.cache/huggingface`  |
| `HF_ACTIVITYNET_FILENAME`            | Optional filename override inside the ActivityNet snapshot     | ``                      |
| `TASK_DATASET_PATH`                  | Optional local validator dataset path                          | ``                      |
| `TASK_SPLIT`                         | Validator task split                                           | `validation`            |
| `REQUIRE_ACCESSIBLE_VIDEOS`          | Skip inaccessible validator task videos                        | `1`                     |
| `TASK_MAX_SAMPLING_ATTEMPTS`         | Max tries to find an accessible validator task                 | `50`                    |
| `VIDEO_AVAILABILITY_CACHE_PATH`      | JSON cache path for validator video availability checks        | ``                      |
| `VIDEO_AVAILABILITY_CACHE_TTL_HOURS` | TTL for cached video availability checks                       | `24`                    |
| `VIDEO_AVAILABILITY_TIMEOUT`         | Timeout for validator-side video availability checks (seconds) | `20`                    |
| `ENABLE_VALIDATOR_API`               | Enable the optional validator `/search` and `/health` API     | `0`                     |
| `VALIDATOR_API_HOST`                 | Host for the optional validator API                            | `0.0.0.0`               |
| `VALIDATOR_API_PORT`                 | Port for the optional validator API                            | `8010`                  |
| `VALIDATOR_API_MAX_MINERS`           | Max miners queried per validator API request                   | `3`                     |
