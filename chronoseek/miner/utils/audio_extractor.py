import os
import tempfile
from dataclasses import dataclass

import av
import numpy as np
import soundfile as sf

from chronoseek.logging import logger


@dataclass
class ExtractedAudio:
    path: str
    duration_sec: float
    cleanup_paths: list[str]


class AudioExtractor:
    """
    Extracts a mono 16kHz WAV track from a downloaded video.

    Decodes in-process via PyAV (links libavcodec/libavformat directly)
    instead of shelling out to the ffmpeg CLI. Chutes TEE instances can lock
    down new OS process spawns once an instance goes hot, which breaks a
    per-request subprocess call even though ffmpeg itself is present in the
    image.
    """

    SAMPLE_RATE = 16000

    @staticmethod
    def extract_audio(video_path: str, timeout: int = 60) -> ExtractedAudio | None:
        fd, output_path = tempfile.mkstemp(prefix="chronoseek-audio-", suffix=".wav")
        os.close(fd)
        cleanup_paths = [output_path]

        try:
            pcm = AudioExtractor._decode_audio_inprocess(video_path)
            if pcm is None or len(pcm) == 0:
                logger.info(
                    "No audio track found; continuing in vision-only mode."
                )
                AudioExtractor.cleanup(
                    ExtractedAudio(
                        path=output_path, duration_sec=0.0, cleanup_paths=cleanup_paths
                    )
                )
                return None

            duration_sec = len(pcm) / float(AudioExtractor.SAMPLE_RATE)
            if duration_sec <= 0:
                logger.warning(
                    "Audio extraction produced an empty or unreadable audio track."
                )
                AudioExtractor.cleanup(
                    ExtractedAudio(
                        path=output_path, duration_sec=0.0, cleanup_paths=cleanup_paths
                    )
                )
                return None

            sf.write(output_path, pcm, AudioExtractor.SAMPLE_RATE)
            return ExtractedAudio(
                path=output_path,
                duration_sec=duration_sec,
                cleanup_paths=cleanup_paths,
            )
        except Exception as exc:
            logger.warning(f"Audio extraction failed or no audio track was present: {exc}")
            AudioExtractor.cleanup(
                ExtractedAudio(
                    path=output_path, duration_sec=0.0, cleanup_paths=cleanup_paths
                )
            )
            return None

    @staticmethod
    def _decode_audio_inprocess(video_path: str) -> np.ndarray | None:
        with av.open(video_path) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                return None
            stream = audio_streams[0]

            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=AudioExtractor.SAMPLE_RATE
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))

        if not chunks:
            return None

        pcm_int16 = np.concatenate(chunks)
        return (pcm_int16.astype(np.float32) / 32768.0)

    @staticmethod
    def cleanup(extracted_audio: ExtractedAudio | None) -> None:
        if extracted_audio is None:
            return

        for cleanup_path in extracted_audio.cleanup_paths:
            try:
                if os.path.exists(cleanup_path):
                    os.remove(cleanup_path)
            except Exception as exc:
                logger.warning(
                    f"Failed to clean up extracted audio artifact {cleanup_path}: {exc}"
                )
