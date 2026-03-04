# ChronoSeek

> **A Bittensor subnet for decentralized semantic video moment retrieval.**

**ChronoSeek** enables semantic search over video content by mapping natural-language scene descriptions to precise timestamp intervals within a video.

---

## ⚠️ MVP Disclaimer (Hackathon)

**Current Status:** Proof of Concept (MVP)

This repository represents the initial **Minimum Viable Product (MVP)** implementation of the ChronoSeek protocol. 

**MVP Scope Limitations:**
*   **Model:** Miners currently use a baseline **CLIP (ViT-B/32)** sliding window approach. This is computationally expensive and not optimized for long-form video.
*   **Dataset:** Validators generate tasks using a fixed subset of the **ActivityNet Captions** dataset (or a fallback public domain set) to ensure deterministic, objective scoring during the bootstrap phase.
*   **Scoring:** Scoring is binary (Pass/Fail) based on a strict Intersection-over-Union (IoU) threshold > 0.5.
*   **Inference:** All inference happens locally on the miner.

**Future Enhancements (Roadmap):**
1.  **Modular Inference:** Integration with **Chutes (SN64)** for serverless, verifiable model execution.
2.  **SOTA Models:** Transition to temporal-aware architectures like **Moment-DETR** or **VideoLlama**.
3.  **Synthetic Tasks:** Implementation of a VLM-based Oracle (using GPT-4o or Gemini) to generate infinite synthetic training tasks from any video URL.
4.  **Vector Caching:** Miners will implement vector databases (Milvus/Chroma) to cache video embeddings, enabling millisecond-level retrieval for repeated queries.

---

## 📚 Project Documentation

This project is organized into the following key documents:

- **[Problem Statement](docs/PROBLEM_STATEMENT.md)**  
  *Why this subnet exists, the "dark data" problem, and the limitations of current search tools.*

- **[System Design](docs/DESIGN.md)**  
  *Technical architecture, including Miner/Validator logic, Synthetic Task Generation, and SOTA research references (CLIP, LLMs).*

- **[Business Logic & Market Rationale](docs/BUSINESS_LOGIC.md)**  
  *Market size ($94B+), commercialization strategy, and competitive advantage against centralized giants.*

---

## 🚀 Quick Start (Hackathon)

### 1. The Core Concept
We are building a decentralized protocol where:
*   **Miners** use AI models (CLIP, Transformers) to "watch" videos and find specific moments.
*   **Validators** generate synthetic queries to grade miners and serve organic requests.
*   **Users** get precise timestamps (e.g., "04:12 - 04:18") for natural language queries.

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
git clone https://github.com/creativeessence/chronoseek-subnet.git
cd chronoseek-subnet

# Install dependencies and create virtualenv
poetry install
```

### 2. Activate Virtual Environment
```bash
poetry env activate
```

### 3. Set up Hugging Face Token
To download models (e.g., CLIP), you need a Hugging Face token.
1. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Set it in your environment:
```bash
export HF_TOKEN=your_token_here
```
*Or add it to your `.env` file.*

## 🏃‍♂️ Running on Testnet (SN298)

### 1. Start the Miner
The miner listens for HTTP requests from validators.
```bash
# Starts miner on port 8000
poetry run python miner.py --netuid 298 --subtensor.network test
```
*Ensure your wallet/hotkey is registered on SN298.*

### 2. Start the Validator
The validator generates synthetic tasks, queries miners, and scores them.
```bash
# Starts validator loop
poetry run python validator.py --netuid 298 --subtensor.network test
```
*Ensure your wallet/hotkey is registered on SN298.*

## ⚙️ Running with PM2 (Production)

For long-running processes, use [PM2](https://pm2.keymetrics.io/).

### 1. Install PM2
```bash
npm install pm2 -g
```

### 2. Start Miner
```bash
# Using the poetry environment python interpreter
pm2 start miner.py --name miner \
    --interpreter $(poetry env info -p)/bin/python \
    -- --netuid 298 --subtensor.network test --wallet.name default --wallet.hotkey default
```

### 3. Start Validator
```bash
# Using the poetry environment python interpreter
pm2 start validator.py --name validator \
    --interpreter $(poetry env info -p)/bin/python \
    -- --netuid 298 --subtensor.network test --wallet.name default --wallet.hotkey default
```

### 4. Manage Processes
```bash
pm2 list
pm2 logs miner
pm2 logs validator
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WALLET_NAME` | Name of your coldkey | `default` |
| `HOTKEY_NAME` | Name of your hotkey | `default` |
| `WALLET_PATH` | Path to your wallet storage | `~/.bittensor/wallets/` |
| `NETUID` | Subnet NetUID | `298` (Mainnet TBD) |
| `NETWORK` | Network (finney, test, local) | `test` |
| `PORT` | Miner HTTP Port | `8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `HF_TOKEN` | Hugging Face Token | `None` |
