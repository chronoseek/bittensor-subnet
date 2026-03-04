import os
import numpy as np
from typing import List, Tuple
from chronoseek.schemas import VideoSearchResult
import bittensor as bt

# Modular components
from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.miner.utils.frame_extractor import FrameExtractor
from chronoseek.miner.utils.clip_engine import CLIPProcessorEngine


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

    def search(self, video_url: str, query: str) -> List[VideoSearchResult]:
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
            return []
        bt.logging.info(f"Video downloaded to {video_path}")

        try:
            # 2. Extract Frames
            bt.logging.info(f">>> Step 2: Extracting Frames (1 fps)")
            frames_data = FrameExtractor.extract_frames(video_path, fps=1)
            if not frames_data:
                bt.logging.error("Frame extraction failed or video is empty.")
                return []
            bt.logging.info(f"Extracted {len(frames_data)} frames.")

            timestamps, images = zip(*frames_data)

            # 3. Inference (Compute Similarity)
            bt.logging.info(f">>> Step 3: Running CLIP Inference")
            probs = self.ml_engine.compute_similarity(query, list(images))
            if len(probs) == 0:
                bt.logging.error("Inference returned no scores.")
                return []
            bt.logging.info(f"Inference complete. Max score: {max(probs):.4f}")

            # 4. Search Heuristics (Thresholding & Merging)
            bt.logging.info(">>> Step 4: Applying Search Heuristics")
            results = self._find_best_segment(probs, timestamps)
            
            if results:
                best = results[0]
                bt.logging.success(f"Best Segment: {best.start:.1f}s - {best.end:.1f}s (Conf: {best.confidence:.4f})")
            
            bt.logging.info("=" * 40)
            return results

        except Exception as e:
            bt.logging.error(f"Search pipeline error: {e}")
            return []
        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)

    def _find_best_segment(
        self, probs: np.ndarray, timestamps: Tuple[float, ...]
    ) -> List[VideoSearchResult]:
        """
        Heuristic to find the best contiguous segment from frame probabilities.
        """
        # Dynamic threshold: Top 10% of scores
        threshold = np.percentile(probs, 90)

        candidates = []
        current_start_idx = None
        gap_tolerance = 2  # frames
        gap_counter = 0

        for i, prob in enumerate(probs):
            if prob > threshold:
                if current_start_idx is None:
                    current_start_idx = i
                gap_counter = 0
            else:
                if current_start_idx is not None:
                    gap_counter += 1
                    if gap_counter > gap_tolerance:
                        end_idx = i - gap_counter
                        segment_probs = probs[current_start_idx : end_idx + 1]
                        score = (
                            np.mean(segment_probs) if len(segment_probs) > 0 else 0.0
                        )
                        candidates.append(
                            (timestamps[current_start_idx], timestamps[end_idx], score)
                        )
                        current_start_idx = None

        # Handle trailing segment
        if current_start_idx is not None:
            end_idx = len(probs) - 1
            segment_probs = probs[current_start_idx : end_idx + 1]
            score = np.mean(segment_probs) if len(segment_probs) > 0 else 0.0
            candidates.append(
                (timestamps[current_start_idx], timestamps[end_idx], score)
            )

        # Rank candidates
        candidates.sort(key=lambda x: x[2], reverse=True)

        results = []
        if candidates:
            best = candidates[0]
            start_t, end_t, score = best

            # Enforce minimum duration (5s)
            if end_t - start_t < 5.0:
                end_t = start_t + 5.0

            results.append(
                VideoSearchResult(start=start_t, end=end_t, confidence=float(score))
            )
        else:
            # Fallback: Middle 10s
            mid = timestamps[len(timestamps) // 2]
            results.append(VideoSearchResult(start=mid, end=mid + 10.0, confidence=0.1))

        return results
