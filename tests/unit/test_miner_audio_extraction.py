import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from chronoseek.miner.utils.audio_extractor import AudioExtractor
from chronoseek.miner.utils.transcript_engine import TranscriptEngine


def _encode_clip(path: str, *, with_audio: bool, duration: float = 2.0) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=320x240:rate=10",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        command += ["-c:v", "libx264", "-c:a", "aac"]
    else:
        command += ["-c:v", "libx264", "-an"]
    command.append(path)
    subprocess.run(command, check=True, capture_output=True)


@pytest.fixture(scope="module")
def clip_with_audio(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("audio-clips") / "with_audio.mp4")
    _encode_clip(path, with_audio=True)
    return path


@pytest.fixture(scope="module")
def clip_without_audio(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("audio-clips") / "no_audio.mp4")
    _encode_clip(path, with_audio=False)
    return path


def test_extract_audio_decodes_in_process_without_subprocess(clip_with_audio, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("extract_audio must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    extracted = AudioExtractor.extract_audio(clip_with_audio)
    try:
        assert extracted is not None
        assert extracted.duration_sec == pytest.approx(2.0, abs=0.2)
        samples, sample_rate = sf.read(extracted.path)
        assert sample_rate == AudioExtractor.SAMPLE_RATE
        assert len(samples) > 0
    finally:
        AudioExtractor.cleanup(extracted)


def test_extract_audio_returns_none_without_audio_track(clip_without_audio):
    extracted = AudioExtractor.extract_audio(clip_without_audio)
    assert extracted is None


def test_extract_audio_returns_none_for_unreadable_file(tmp_path):
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"not a real video")

    extracted = AudioExtractor.extract_audio(str(bogus))
    assert extracted is None


def test_transcribe_feeds_raw_array_not_a_file_path(tmp_path):
    engine = TranscriptEngine.__new__(TranscriptEngine)
    engine._disabled = False

    stub_pipeline = MagicMock(return_value={"text": "hello world", "chunks": []})
    engine._pipeline = stub_pipeline

    wav_path = str(tmp_path / "audio.wav")
    samples = np.zeros(16000, dtype=np.float32)
    sf.write(wav_path, samples, 16000)

    engine.transcribe(wav_path, audio_duration_sec=1.0)

    assert stub_pipeline.call_count == 1
    call_args = stub_pipeline.call_args[0]
    payload = call_args[0]
    assert isinstance(payload, dict)
    assert "raw" in payload and "sampling_rate" in payload
    assert payload["sampling_rate"] == 16000
    assert isinstance(payload["raw"], np.ndarray)


def test_transcribe_returns_empty_when_audio_file_unreadable():
    engine = TranscriptEngine.__new__(TranscriptEngine)
    engine._disabled = False
    engine._pipeline = MagicMock()

    segments = engine.transcribe("/nonexistent/path.wav", audio_duration_sec=1.0)

    assert segments == []
    engine._pipeline.assert_not_called()
