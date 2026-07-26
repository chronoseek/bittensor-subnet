# ChronoSeek Subnet Validator Docker Image
# Build: docker build -t chronoseek-validator .
# Run: docker compose up -d

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml ./

# Install uv for faster dependency resolution
RUN pip install --no-cache-dir --upgrade pip uv

# Copy application code
COPY . .

# Install dependencies
RUN uv pip install --system --no-cache -e .

# Create non-root user for security
RUN useradd -m -u 1000 validator && \
    chown -R validator:validator /app

# Environment configuration
ENV NETWORK=finney
ENV NETUID=20
ENV WALLET_NAME=validator
ENV HOTKEY_NAME=default
ENV LOG_LEVEL=INFO

# Run the validator
CMD ["python", "validator.py"]
