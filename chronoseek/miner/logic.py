import os
import numpy as np
from typing import List, Tuple
from chronoseek.protocol_models import VideoSearchResult
import bittensor as bt

# Modular components
from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.miner.utils.frame_extractor import FrameExtractor
from chronoseek.miner.utils.clip_engine import CLIPProcessorEngine


class SearchPipelineError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class MinerLogic:
    """
    Core logic for the ChronoSeek Miner.

    This implementation follows a modular design pattern:
    1. VideoDownloader: Handles fetching content.
    2. FrameExtractor: Handles video processing.
    3. CLIPProcessorEngine: Handles ML inference.
    4. MinerLogic: Orchestrates the pipeline and implements search heuristics.
    """

    def __init__(self):
        # Initialize ML engine once at startup
        self.ml_engine = CLIPProcessorEngine(model_id="openai/clip-vit-base-patch32")

    def search(self, video_url: str, query: str, top_k: int = 5) -> List[VideoSearchResult]:
        """
        Execute the search pipeline.
        """
        bt.logging.info(f"Processing query: '{query}' for video: {video_url}")

        # 1. Download
        bt.logging.info("=" * 40)
        bt.logging.info(f"MINER: STARTING SEARCH")
        bt.logging.info(f"Video: {video_url}")
        bt.logging.info(f"Query: {query}")
        bt.logging.info("=" * 40)

        bt.logging.info(f">>> Step 1: Downloading Video")
        video_path = VideoDownloader.download_video(video_url)
        if not video_path:
            bt.logging.error("Video download failed.")
            raise SearchPipelineError(
                "VIDEO_FETCH_FAILED",
                "The video URL could not be fetched.",
                {"video_url": video_url},
            )
        bt.logging.info(f"Video downloaded to {video_path}")

        try:
            # 2. Extract Frames
            bt.logging.info(f">>> Step 2: Extracting Frames (1 fps)")
            frames_data = FrameExtractor.extract_frames(video_path, fps=1)
            if not frames_data:
                bt.logging.error("Frame extraction failed or video is empty.")
                raise SearchPipelineError(
                    "VIDEO_UNREADABLE",
                    "The downloaded video could not be decoded into frames.",
                    {"video_url": video_url},
                )
            bt.logging.info(f"Extracted {len(frames_data)} frames.")

            timestamps, images = zip(*frames_data)

            # 3. Inference (Compute Similarity)
            bt.logging.info(f">>> Step 3: Running CLIP Inference")
            probs = self.ml_engine.compute_similarity(query, list(images))
            if len(probs) == 0:
                bt.logging.error("Inference returned no scores.")
                raise SearchPipelineError(
                    "INFERENCE_FAILED",
                    "The miner could not compute similarity scores for this request.",
                    {"video_url": video_url},
                )
            bt.logging.info(f"Inference complete. Max score: {max(probs):.4f}")

            # 4. Search Heuristics (Thresholding & Merging)
            bt.logging.info(">>> Step 4: Applying Search Heuristics")
            results = self._find_best_segment(probs, timestamps, top_k=top_k)
            
            if results:
                best = results[0]
                bt.logging.success(f"Best Segment: {best.start:.1f}s - {best.end:.1f}s (Conf: {best.confidence:.4f})")
            
            bt.logging.info("=" * 40)
            return results

        except SearchPipelineError:
            raise
        except Exception as e:
            bt.logging.error(f"Search pipeline error: {e}")
            raise SearchPipelineError(
                "INTERNAL_ERROR",
                "The miner encountered an unexpected search pipeline error.",
                {"video_url": video_url},
            ) from e
        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)

    def _interval_iou(self, left: tuple[float, float], right: tuple[float, float]) -> float:
        start = max(left[0], right[0])
        end = min(left[1], right[1])
        intersection = max(0.0, end - start)
        union = max(left[1], right[1]) - min(left[0], right[0])
        if union <= 0:
            return 0.0
        return intersection / union

    def _find_best_segment(
        self, probs: np.ndarray, timestamps: Tuple[float, ...], top_k: int = 5
    ) -> List[VideoSearchResult]:
        """
        Find top-k temporal segments from CLIP frame similarities.
        """
        if len(probs) == 0:
            return []

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        smoothed = np.convolve(probs, kernel, mode="same")
        threshold = max(
            float(np.percentile(smoothed, 75)),
            float(np.mean(smoothed) + 0.35 * np.std(smoothed)),
        )
        high_conf_threshold = float(np.percentile(probs, 85))

        candidates = []
        current_start_idx = None
        gap_tolerance = 1
        gap_counter = 0

        for i, score in enumerate(smoothed):
            is_active = score >= threshold or probs[i] >= high_conf_threshold
            if is_active:
                if current_start_idx is None:
                    current_start_idx = i
                gap_counter = 0
            else:
                if current_start_idx is not None:
                    gap_counter += 1
                    if gap_counter > gap_tolerance:
                        end_idx = i - gap_counter
                        segment_probs = smoothed[current_start_idx : end_idx + 1]
                        raw_probs = probs[current_start_idx : end_idx + 1]
                        score = (
                            0.65 * float(np.max(segment_probs))
                            + 0.35 * float(np.mean(raw_probs))
                            if len(segment_probs) > 0
                            else 0.0
                        )
                        start_idx = max(0, current_start_idx - 1)
                        end_idx = min(len(timestamps) - 1, end_idx + 1)
                        candidates.append(
                            (timestamps[start_idx], timestamps[end_idx], score)
                        )
                        current_start_idx = None

        if current_start_idx is not None:
            end_idx = len(probs) - 1
            segment_probs = smoothed[current_start_idx : end_idx + 1]
            raw_probs = probs[current_start_idx : end_idx + 1]
            score = (
                0.65 * float(np.max(segment_probs))
                + 0.35 * float(np.mean(raw_probs))
                if len(segment_probs) > 0
                else 0.0
            )
            start_idx = max(0, current_start_idx - 1)
            candidates.append(
                (timestamps[start_idx], timestamps[end_idx], score)
            )

        if not candidates:
            ranked_frames = np.argsort(smoothed)[::-1][: max(1, min(top_k, len(smoothed)))]
            frame_duration = (
                float(np.median(np.diff(np.array(timestamps)))) if len(timestamps) > 1 else 1.0
            )
            default_window = max(4.0, 4 * frame_duration)
            for idx in ranked_frames:
                center = timestamps[int(idx)]
                candidates.append(
                    (
                        max(0.0, center - default_window / 2),
                        center + default_window / 2,
                        float(smoothed[int(idx)]),
                    )
                )

        candidates.sort(key=lambda x: x[2], reverse=True)

        results = []
        for start_t, end_t, score in candidates:
            if end_t - start_t < 5.0:
                end_t = start_t + 5.0
            interval = (float(start_t), float(end_t))
            if any(self._interval_iou(interval, (res.start, res.end)) > 0.7 for res in results):
                continue
            results.append(
                VideoSearchResult(
                    start=interval[0],
                    end=interval[1],
                    confidence=float(max(0.0, min(1.0, score))),
                )
            )
            if len(results) >= top_k:
                break

        if not results:
            mid = float(timestamps[len(timestamps) // 2])
            results.append(VideoSearchResult(start=mid, end=mid + 8.0, confidence=0.1))

        return results
