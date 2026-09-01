import os
from dataclasses import dataclass
from typing import List

import soundfile as sf
import torch

from chronoseek.logging import logger


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptEngine:
    """
    Wraps a lightweight timestamped ASR pipeline.
    Falls back to vision-only mode when the ASR model is unavailable.
    """

    def __init__(self, model_id: str = "openai/whisper-tiny"):
        self.model_id = model_id
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipeline = None
        self._disabled = False
        self._ensure_pipeline()

    def _ensure_pipeline(self):
        if self._disabled:
            return None
        if self._pipeline is not None:
            return self._pipeline

        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning(
                "HF_TOKEN not found in environment. Transcript model downloads may be rate-limited."
            )

        try:
            from transformers import pipeline

            logger.info(
                f"Loading transcript model '{self.model_id}' on "
                f"{'cuda' if self.device >= 0 else 'cpu'}..."
            )
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                token=token,
                device=self.device,
                chunk_length_s=30,
            )
            logger.success("Transcript model loaded successfully.")
        except Exception as exc:
            self._disabled = True
            logger.warning(
                f"Transcript model unavailable; continuing in vision-only mode: {exc}"
            )
            return None

        return self._pipeline

    def transcribe(
        self, audio_path: str, audio_duration_sec: float | None = None
    ) -> List[TranscriptSegment]:
        asr_pipeline = self._ensure_pipeline()
        if asr_pipeline is None:
            return []

        try:
            samples, sample_rate = sf.read(audio_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
        except Exception as exc:
            logger.warning(f"Failed to read extracted audio for transcription: {exc}")
            return []

        try:
            # Feed a raw array rather than a file path: the pipeline shells
            # out to the ffmpeg CLI internally (ffmpeg_read) for path inputs,
            # which is exactly the per-request subprocess spawn this runtime
            # otherwise avoids - see AudioExtractor's docstring.
            result = asr_pipeline(
                {"raw": samples, "sampling_rate": sample_rate},
                return_timestamps=True,
            )
        except Exception as exc:
            logger.warning(f"Transcript generation failed: {exc}")
            return []

        segments = self._parse_segments(result, audio_duration_sec)
        if not segments:
            logger.info("Transcript generation returned no timestamped segments.")
        return segments

    @staticmethod
    def _parse_segments(
        result, audio_duration_sec: float | None
    ) -> List[TranscriptSegment]:
        chunks = result.get("chunks") if isinstance(result, dict) else None
        if chunks:
            segments: List[TranscriptSegment] = []
            for chunk in chunks:
                raw_text = str(chunk.get("text", "")).strip()
                timestamp = chunk.get("timestamp")
                if (
                    not raw_text
                    or not isinstance(timestamp, (list, tuple))
                    or len(timestamp) != 2
                ):
                    continue
                start, end = timestamp
                if start is None:
                    continue
                start_f = float(start)
                end_f = (
                    float(end)
                    if end is not None
                    else float(audio_duration_sec or start_f)
                )
                if end_f <= start_f:
                    continue
                segments.append(
                    TranscriptSegment(start=start_f, end=end_f, text=raw_text)
                )
            return segments

        if isinstance(result, dict):
            raw_text = str(result.get("text", "")).strip()
            if raw_text and audio_duration_sec and audio_duration_sec > 0:
                return [
                    TranscriptSegment(
                        start=0.0,
                        end=float(audio_duration_sec),
                        text=raw_text,
                    )
                ]

        return []
