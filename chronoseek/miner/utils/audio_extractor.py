import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass

import bittensor as bt


@dataclass
class ExtractedAudio:
    path: str
    duration_sec: float
    cleanup_paths: list[str]


class AudioExtractor:
    """
    Extracts a mono 16kHz WAV track from a downloaded video.
    """

    @staticmethod
    def extract_audio(video_path: str, timeout: int = 60) -> ExtractedAudio | None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            bt.logging.warning("ffmpeg is not available. Skipping audio extraction.")
            return None

        fd, output_path = tempfile.mkstemp(prefix="chronoseek-audio-", suffix=".wav")
        os.close(fd)
        cleanup_paths = [output_path]

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            output_path,
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                bt.logging.warning(
                    f"Audio extraction failed or no audio track was present: {stderr or 'ffmpeg returned non-zero exit status.'}"
                )
                AudioExtractor.cleanup(
                    ExtractedAudio(
                        path=output_path,
                        duration_sec=0.0,
                        cleanup_paths=cleanup_paths,
                    )
                )
                return None

            duration_sec = AudioExtractor._get_wav_duration(output_path)
            if duration_sec <= 0:
                bt.logging.warning(
                    "Audio extraction produced an empty or unreadable WAV file."
                )
                AudioExtractor.cleanup(
                    ExtractedAudio(
                        path=output_path,
                        duration_sec=0.0,
                        cleanup_paths=cleanup_paths,
                    )
                )
                return None

            return ExtractedAudio(
                path=output_path,
                duration_sec=duration_sec,
                cleanup_paths=cleanup_paths,
            )
        except Exception as exc:
            bt.logging.warning(f"Audio extraction failed: {exc}")
            AudioExtractor.cleanup(
                ExtractedAudio(
                    path=output_path,
                    duration_sec=0.0,
                    cleanup_paths=cleanup_paths,
                )
            )
            return None

    @staticmethod
    def _get_wav_duration(audio_path: str) -> float:
        try:
            with wave.open(audio_path, "rb") as handle:
                frame_rate = handle.getframerate()
                frame_count = handle.getnframes()
                if frame_rate <= 0:
                    return 0.0
                return float(frame_count) / float(frame_rate)
        except Exception:
            return 0.0

    @staticmethod
    def cleanup(extracted_audio: ExtractedAudio | None) -> None:
        if extracted_audio is None:
            return

        for cleanup_path in extracted_audio.cleanup_paths:
            try:
                if os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
            except Exception as exc:
                bt.logging.warning(
                    f"Failed to clean up extracted audio artifact {cleanup_path}: {exc}"
                )
