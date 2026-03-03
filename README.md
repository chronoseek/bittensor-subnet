# Semantic Video Moment Retrieval (SVMR) Subnet

> **A Bittensor subnet for decentralized semantic video moment retrieval.**

The **SVMR Subnet** enables semantic search over video content by mapping natural-language scene descriptions to precise timestamp intervals within a video.

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

## 🏃‍♂️ Running Locally

### 1. Start the Miner
The miner listens for HTTP requests from validators.
```bash
# Starts miner on port 8000
poetry run python miner.py
```
*You can configure the wallet/hotkey via environment variables (see `miner.py`).*

### 2. Start the Validator
The validator generates synthetic tasks, queries miners, and scores them.
```bash
# Starts validator loop
poetry run python validator.py
```
*Note: For local testing without a running Bittensor chain, ensure the code handles mock metagraphs appropriately.*

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WALLET_NAME` | Name of your coldkey | `default` |
| `HOTKEY_NAME` | Name of your hotkey | `default` |
| `NETUID` | Subnet NetUID | `1` |
| `NETWORK` | Network (finney, test, local) | `finney` |
| `PORT` | Miner HTTP Port | `8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
