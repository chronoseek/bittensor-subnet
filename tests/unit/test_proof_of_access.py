import subprocess

import pytest

from chronoseek.video.proof_of_access import (
    compute_proof_of_access_hash,
    proof_of_access_hashes_match,
    sample_frame_timestamps,
)


def _encode_test_clip(path: str, *, source: str, crf: int, size: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{source}=duration=2:size={size}:rate=10",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            path,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def high_quality_clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "hq.mp4")
    _encode_test_clip(path, source="testsrc", crf=18, size="640x480")
    return path


@pytest.fixture(scope="module")
def low_quality_same_content_clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "lq.mp4")
    _encode_test_clip(path, source="testsrc", crf=35, size="320x240")
    return path


@pytest.fixture(scope="module")
def different_content_clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "different.mp4")
    _encode_test_clip(path, source="smptebars", crf=18, size="640x480")
    return path


def test_sample_frame_timestamps_covers_first_middle_last():
    timestamps = sample_frame_timestamps(10.0)
    assert timestamps[0] == 0.0
    assert timestamps[1] == 5.0
    assert 9.5 <= timestamps[2] < 10.0


def test_sample_frame_timestamps_handles_zero_duration():
    assert sample_frame_timestamps(0.0) == [0.0]


def test_compute_proof_of_access_hash_returns_none_for_unreadable_file(tmp_path):
    bad_path = tmp_path / "not_a_video.mp4"
    bad_path.write_bytes(b"not a real video")

    assert compute_proof_of_access_hash(str(bad_path)) is None


def test_same_content_matches_despite_quality_and_resolution_differences(
    high_quality_clip, low_quality_same_content_clip
):
    """Regression test: a plain cryptographic hash of decoded (even
    resized+grayscaled) frames false-fails here, because two honest
    downloads of the same source can legitimately differ in resolution
    and compression quality (e.g. YouTube serving different encodes)."""
    hash_a = compute_proof_of_access_hash(high_quality_clip)
    hash_b = compute_proof_of_access_hash(low_quality_same_content_clip)

    assert hash_a is not None
    assert hash_b is not None
    assert proof_of_access_hashes_match(hash_a, hash_b)


def test_different_content_does_not_match(high_quality_clip, different_content_clip):
    hash_a = compute_proof_of_access_hash(high_quality_clip)
    hash_b = compute_proof_of_access_hash(different_content_clip)

    assert hash_a is not None
    assert hash_b is not None
    assert not proof_of_access_hashes_match(hash_a, hash_b)


def test_hashes_match_requires_equal_frame_counts():
    assert not proof_of_access_hashes_match("a" * 16, "a" * 32)
