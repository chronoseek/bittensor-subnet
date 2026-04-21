import os
from dataclasses import dataclass
from typing import List

import bittensor as bt
import torch


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
            bt.logging.warning(
                "HF_TOKEN not found in environment. Transcript model downloads may be rate-limited."
            )

        try:
            from transformers import pipeline

            bt.logging.info(
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
            bt.logging.success("Transcript model loaded successfully.")
        except Exception as exc:
            self._disabled = True
            bt.logging.warning(
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
            result = asr_pipeline(audio_path, return_timestamps=True)
        except Exception as exc:
            bt.logging.warning(f"Transcript generation failed: {exc}")
            return []

        segments = self._parse_segments(result, audio_duration_sec)
        if not segments:
            bt.logging.info("Transcript generation returned no timestamped segments.")
        return segments

    @staticmethod
    def _parse_segments(result, audio_duration_sec: float | None) -> List[TranscriptSegment]:
        chunks = result.get("chunks") if isinstance(result, dict) else None
        if chunks:
            segments: List[TranscriptSegment] = []
            for chunk in chunks:
                raw_text = str(chunk.get("text", "")).strip()
                timestamp = chunk.get("timestamp")
                if not raw_text or not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
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
