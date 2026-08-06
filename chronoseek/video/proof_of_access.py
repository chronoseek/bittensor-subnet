"""Shared proof-of-access hashing.

Confirms a peer actually downloaded and decoded a given video, without
requiring byte-identical files: codec/container/bitrate/resolution can
legitimately differ between two honest downloads of the same URL (e.g.
YouTube serving a different encode to each request). A cryptographic hash
of decoded pixels is still far too sensitive to this - even after
resizing and grayscaling, two honest downloads of the same source at
different quality settings hash completely differently. This uses a
perceptual hash (average-hash) per sampled frame instead, compared by
Hamming distance under a threshold rather than exact equality, so
compression/resolution noise doesn't cause a false mismatch while
genuinely different content still does.
"""

import cv2

FRAME_HASH_GRID_SIZE = 8
FRAME_HASH_BITS = FRAME_HASH_GRID_SIZE * FRAME_HASH_GRID_SIZE
# Default distance budget spent per sampled frame - calibrated empirically
# against real re-encodes of the same source at very different
# quality/resolution (see tests/unit/test_proof_of_access.py).
DEFAULT_MAX_HAMMING_DISTANCE_PER_FRAME = 10


def sample_frame_timestamps(duration_seconds: float) -> list[float]:
    """First, middle, and last frame timestamps (seconds) for a video."""
    if duration_seconds <= 0:
        return [0.0]
    last = max(0.0, duration_seconds - 0.1)
    return [0.0, duration_seconds / 2.0, last]


def _average_hash_bits(frame) -> int:
    """8x8 average-hash of a single decoded frame, as a 64-bit integer."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (FRAME_HASH_GRID_SIZE, FRAME_HASH_GRID_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    mean = small.mean()
    value = 0
    for pixel in small.flatten():
        value = (value << 1) | int(pixel > mean)
    return value


def compute_proof_of_access_hash(video_path: str) -> str | None:
    """
    Perceptual hash of decoded frames at fixed timestamps (first/middle/last).

    Returns a hex string encoding one 64-bit average-hash per sampled
    frame, concatenated in timestamp order. Returns None if no frame could
    be decoded at all.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

        frame_hashes: list[int] = []
        for timestamp in sample_frame_timestamps(duration):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_hashes.append(_average_hash_bits(frame))

        if not frame_hashes:
            return None
        return "".join(format(bits, "016x") for bits in frame_hashes)
    finally:
        cap.release()


def proof_of_access_hashes_match(
    hash_a: str,
    hash_b: str,
    *,
    max_hamming_distance_per_frame: int = DEFAULT_MAX_HAMMING_DISTANCE_PER_FRAME,
) -> bool:
    """
    Whether two `compute_proof_of_access_hash` outputs represent the same
    underlying video content, allowing per-frame bit differences up to
    `max_hamming_distance_per_frame` (out of 64 bits/frame) to absorb
    compression/resolution noise between two honest downloads.
    """
    if len(hash_a) != len(hash_b) or len(hash_a) % 16 != 0:
        return False

    frame_count = len(hash_a) // 16
    max_total_distance = max_hamming_distance_per_frame * frame_count

    total_distance = 0
    for i in range(frame_count):
        chunk_a = int(hash_a[i * 16 : (i + 1) * 16], 16)
        chunk_b = int(hash_b[i * 16 : (i + 1) * 16], 16)
        total_distance += bin(chunk_a ^ chunk_b).count("1")

    return total_distance <= max_total_distance
