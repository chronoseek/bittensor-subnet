import os
import time
import numpy as np
from typing import List, Sequence, Tuple
from chronoseek.protocol_models import VideoSearchResult
import bittensor as bt

# Modular components
from chronoseek.miner.utils.audio_extractor import AudioExtractor
from chronoseek.miner.utils.video_downloader import VideoDownloader
from chronoseek.miner.utils.frame_extractor import FrameExtractor
from chronoseek.miner.utils.clip_engine import CLIPProcessorEngine
from chronoseek.miner.utils.transcript_engine import TranscriptEngine, TranscriptSegment


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

    Retrieval uses two-stage temporal refinement: coarse uniform sampling, CLIP scoring,
    peak-based windows, then higher-FPS sampling inside those windows before segment search.
    """

    COARSE_FPS = 1
    REFINE_FPS = 4
    REFINE_WINDOW_PAD_SEC = 3.0
    REFINE_MIN_PEAK_SEP_SEC = 4.0
    REFINE_MAX_WINDOWS = 3
    LONG_VIDEO_SEC = 300.0
    MEDIUM_VIDEO_SEC = 60.0
    MEDIUM_COARSE_FPS = 0.5
    LONG_COARSE_FPS = 0.25
    REFINE_SKIP_PEAK_MIN = 0.84
    REFINE_SKIP_MARGIN_MIN = 0.18
    REFINE_SKIP_P90_GAP_MIN = 0.15
    AUDIO_BOOST_FACTOR = 0.25

    def __init__(self):
        # Initialize ML engine once at startup
        self.ml_engine = CLIPProcessorEngine(model_id="openai/clip-vit-base-patch32")
        self.transcript_engine = TranscriptEngine(model_id="openai/whisper-tiny")

    def search(
        self, video_url: str, query: str, top_k: int = 5
    ) -> List[VideoSearchResult]:
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
        request_t0 = time.perf_counter()
        download_sec = 0.0
        coarse_extract_sec = 0.0
        coarse_infer_sec = 0.0
        fine_extract_sec = 0.0
        fine_infer_sec = 0.0
        audio_extract_sec = 0.0
        transcript_sec = 0.0
        heuristic_sec = 0.0

        bt.logging.info(f">>> Step 1: Downloading Video")
        t0 = time.perf_counter()
        downloaded_video = VideoDownloader.download_video(video_url)
        download_sec = time.perf_counter() - t0
        if not downloaded_video:
            bt.logging.error("Video download failed.")
            raise SearchPipelineError(
                "VIDEO_FETCH_FAILED",
                "The video URL could not be fetched.",
                {"video_url": video_url},
            )
        video_path = downloaded_video.path
        bt.logging.info(
            f"Video downloaded to {video_path} (download: {download_sec:.2f}s)"
        )

        try:
            # 2. Coarse frames + CLIP
            coarse_fps = float(self.COARSE_FPS)
            bt.logging.info(
                f">>> Step 2: Coarse frame extraction ({coarse_fps:g} fps)"
            )
            t0 = time.perf_counter()
            frames_data = FrameExtractor.extract_frames(
                video_path, fps=coarse_fps
            )
            if not frames_data:
                bt.logging.error("Frame extraction failed or video is empty.")
                raise SearchPipelineError(
                    "VIDEO_UNREADABLE",
                    "The downloaded video could not be decoded into frames.",
                    {"video_url": video_url},
                )
            bt.logging.info(f"Coarse: extracted {len(frames_data)} frames.")

            coarse_ts, images = zip(*frames_data)
            coarse_ts_arr = np.array(coarse_ts, dtype=np.float64)
            coarse_ts_tuple = tuple(float(t) for t in coarse_ts)
            video_duration_sec = float(coarse_ts_arr[-1]) if len(coarse_ts_arr) else 0.0

            adaptive_fps = self._select_coarse_fps(video_duration_sec)
            if adaptive_fps < coarse_fps:
                bt.logging.info(
                    f"Coarse duration={video_duration_sec:.1f}s; resampling coarse pass "
                    f"at {adaptive_fps:g} fps for faster inference."
                )
                frames_data = FrameExtractor.extract_frames(video_path, fps=adaptive_fps)
                if frames_data:
                    coarse_ts, images = zip(*frames_data)
                    coarse_ts_arr = np.array(coarse_ts, dtype=np.float64)
                    coarse_ts_tuple = tuple(float(t) for t in coarse_ts)
                else:
                    bt.logging.warning(
                        "Adaptive coarse resample returned no frames; using initial coarse set."
                    )
            coarse_extract_sec = time.perf_counter() - t0
            bt.logging.info(
                f"Coarse frames ready: {len(coarse_ts_tuple)} points "
                f"(duration: {video_duration_sec:.1f}s, extract: {coarse_extract_sec:.2f}s)"
            )

            # 3. Coarse inference
            bt.logging.info(">>> Step 3a: CLIP inference (coarse)")
            t0 = time.perf_counter()
            coarse_probs = self.ml_engine.compute_similarity(query, list(images))
            coarse_infer_sec = time.perf_counter() - t0
            if len(coarse_probs) == 0:
                bt.logging.error("Inference returned no scores.")
                raise SearchPipelineError(
                    "INFERENCE_FAILED",
                    "The miner could not compute similarity scores for this request.",
                    {"video_url": video_url},
                )
            bt.logging.info(
                f"Coarse max score: {float(np.max(coarse_probs)):.4f} "
                f"(infer: {coarse_infer_sec:.2f}s)"
            )

            # 4. Fine windows + second pass (adaptive)
            video_end = float(coarse_ts_arr[-1]) if len(coarse_ts_arr) else 0.0
            if self._should_skip_refine(coarse_probs):
                bt.logging.info(">>> Step 3b: skipping refine pass (strong coarse confidence)")
                refine_windows = []
            else:
                num_refine = self._refine_window_budget(top_k)
                refine_windows = self._pick_refine_windows(
                    coarse_probs,
                    coarse_ts_tuple,
                    num_windows=num_refine,
                    pad_sec=self.REFINE_WINDOW_PAD_SEC,
                    min_peak_sep_sec=self.REFINE_MIN_PEAK_SEP_SEC,
                    video_end=video_end,
                )

            merged_ts = coarse_ts_tuple
            merged_probs = coarse_probs

            if refine_windows:
                bt.logging.info(
                    f">>> Step 3b: Refining {len(refine_windows)} temporal windows "
                    f"at ~{self.REFINE_FPS} fps"
                )
                t0 = time.perf_counter()
                fine_data = FrameExtractor.extract_frames_in_windows(
                    video_path,
                    refine_windows,
                    fps=float(self.REFINE_FPS),
                )
                fine_extract_sec = time.perf_counter() - t0
                if fine_data:
                    fine_ts, fine_imgs = zip(*fine_data)
                    t0 = time.perf_counter()
                    fine_probs = self.ml_engine.compute_similarity(
                        query, list(fine_imgs)
                    )
                    fine_infer_sec = time.perf_counter() - t0
                    if len(fine_probs) > 0:
                        merged_ts, merged_probs = self._merge_coarse_fine_timeline(
                            coarse_ts_tuple,
                            coarse_probs,
                            list(fine_ts),
                            fine_probs,
                            refine_windows,
                        )
                        bt.logging.info(
                            f"Merged timeline: {len(merged_ts)} points "
                            f"(max {float(np.max(merged_probs)):.4f}, "
                            f"fine_extract: {fine_extract_sec:.2f}s, "
                            f"fine_infer: {fine_infer_sec:.2f}s)"
                        )
                    else:
                        bt.logging.warning(
                            "Fine pass produced no scores; using coarse only."
                        )
                else:
                    bt.logging.warning("Fine extraction empty; using coarse only.")

            # 5. Audio extraction + transcript scoring
            bt.logging.info(">>> Step 4: Extracting audio")
            t0 = time.perf_counter()
            extracted_audio = AudioExtractor.extract_audio(video_path)
            audio_extract_sec = time.perf_counter() - t0

            fused_ts = merged_ts
            fused_probs = merged_probs
            fusion_mode = "vision_only"

            if extracted_audio is None:
                bt.logging.info(
                    f"Audio unavailable; using vision-only scoring (audio_extract: {audio_extract_sec:.2f}s)"
                )
            else:
                bt.logging.info(
                    f"Audio extracted to {extracted_audio.path} "
                    f"(duration: {extracted_audio.duration_sec:.1f}s, extract: {audio_extract_sec:.2f}s)"
                )
                bt.logging.info(">>> Step 5: Generating transcript")
                t0 = time.perf_counter()
                transcript_segments = self.transcript_engine.transcribe(
                    extracted_audio.path,
                    audio_duration_sec=extracted_audio.duration_sec,
                )
                transcript_sec = time.perf_counter() - t0
                scored_audio_segments = self._score_transcript_segments(
                    query, transcript_segments
                )
                if scored_audio_segments:
                    fused_ts, fused_probs, fusion_mode = self._fuse_visual_audio_timeline(
                        merged_ts,
                        merged_probs,
                        scored_audio_segments,
                    )
                    max_audio_score = max(score for _, _, score, _ in scored_audio_segments)
                    bt.logging.info(
                        f"Transcript segments: {len(transcript_segments)} "
                        f"(transcribe: {transcript_sec:.2f}s, max_audio_score: {max_audio_score:.4f})"
                    )
                    bt.logging.info(
                        "Fusion mode: vision_audio "
                        f"(vision primary, audio boost factor={self.AUDIO_BOOST_FACTOR:.2f})"
                    )
                else:
                    bt.logging.info(
                        f"No usable audio-derived timeline; using vision-only scoring "
                        f"(transcribe: {transcript_sec:.2f}s)"
                    )

            # 6. Search Heuristics (Thresholding & Merging)
            bt.logging.info(">>> Step 6: Applying search heuristics")
            t0 = time.perf_counter()
            results = self._find_best_segment(
                fused_probs, fused_ts, top_k=top_k
            )
            heuristic_sec = time.perf_counter() - t0
            
            if results:
                best = results[0]
                bt.logging.success(f"Best Segment: {best.start:.1f}s - {best.end:.1f}s (Conf: {best.confidence:.4f})")
            total_sec = time.perf_counter() - request_t0
            bt.logging.info(
                "Timing summary: "
                f"download={download_sec:.2f}s, "
                f"coarse_extract={coarse_extract_sec:.2f}s, "
                f"coarse_infer={coarse_infer_sec:.2f}s, "
                f"fine_extract={fine_extract_sec:.2f}s, "
                f"fine_infer={fine_infer_sec:.2f}s, "
                f"audio_extract={audio_extract_sec:.2f}s, "
                f"transcript={transcript_sec:.2f}s, "
                f"heuristics={heuristic_sec:.2f}s, "
                f"total={total_sec:.2f}s"
            )
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
            AudioExtractor.cleanup(
                extracted_audio if "extracted_audio" in locals() else None
            )
            VideoDownloader.cleanup(
                downloaded_video if "downloaded_video" in locals() else None
            )

    @staticmethod
    def _pick_refine_windows(
        probs: np.ndarray,
        timestamps: Tuple[float, ...],
        num_windows: int,
        pad_sec: float,
        min_peak_sep_sec: float,
        video_end: float,
    ) -> List[Tuple[float, float]]:
        """Peaks on smoothed coarse scores → padded [start, end] seconds for dense sampling."""
        if len(probs) == 0 or len(timestamps) == 0:
            return []

        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        smoothed = np.convolve(probs.astype(np.float32), kernel, mode="same")
        order = np.argsort(smoothed)[::-1]

        picked: List[float] = []
        for idx in order:
            t = float(timestamps[int(idx)])
            if len(picked) >= num_windows:
                break
            if all(abs(t - c) >= min_peak_sep_sec for c in picked):
                picked.append(t)

        if not picked:
            picked = [float(timestamps[int(np.argmax(smoothed))])]

        end_limit = max(
            float(video_end),
            float(max(timestamps)) if timestamps else 0.0,
        )
        return [
            (max(0.0, t - pad_sec), min(end_limit, t + pad_sec)) for t in picked
        ]

    def _select_coarse_fps(self, video_duration_sec: float) -> float:
        if video_duration_sec >= self.LONG_VIDEO_SEC:
            return self.LONG_COARSE_FPS
        if video_duration_sec >= self.MEDIUM_VIDEO_SEC:
            return self.MEDIUM_COARSE_FPS
        return float(self.COARSE_FPS)

    def _refine_window_budget(self, top_k: int) -> int:
        return max(1, min(self.REFINE_MAX_WINDOWS, max(2, top_k)))

    def _should_skip_refine(self, coarse_probs: np.ndarray) -> bool:
        if len(coarse_probs) < 3:
            return False
        sorted_scores = np.sort(coarse_probs.astype(np.float32))[::-1]
        top1 = float(sorted_scores[0])
        top2 = float(sorted_scores[1])
        p90 = float(np.percentile(sorted_scores, 90))
        margin = top1 - top2
        p90_gap = top1 - p90
        return (
            top1 >= self.REFINE_SKIP_PEAK_MIN
            and margin >= self.REFINE_SKIP_MARGIN_MIN
            and p90_gap >= self.REFINE_SKIP_P90_GAP_MIN
        )

    @staticmethod
    def _merge_coarse_fine_timeline(
        coarse_ts: Tuple[float, ...],
        coarse_probs: np.ndarray,
        fine_ts: List[float],
        fine_probs: np.ndarray,
        windows: Sequence[Tuple[float, float]],
    ) -> Tuple[Tuple[float, ...], np.ndarray]:
        """Replace coarse points inside refine windows with higher-FPS samples."""
        if len(fine_ts) == 0 or len(fine_probs) == 0:
            return coarse_ts, coarse_probs

        def in_any_window(t: float) -> bool:
            return any(w0 - 1e-3 <= t <= w1 + 1e-3 for w0, w1 in windows)

        kept = [
            (float(t), float(p))
            for t, p in zip(coarse_ts, coarse_probs)
            if not in_any_window(float(t))
        ]
        fine_pairs = [(float(t), float(p)) for t, p in zip(fine_ts, fine_probs)]
        merged = sorted(kept + fine_pairs, key=lambda x: x[0])
        if not merged:
            return coarse_ts, coarse_probs

        ts, ps = zip(*merged)
        return tuple(ts), np.array(ps, dtype=np.float32)

    def _score_transcript_segments(
        self,
        query: str,
        transcript_segments: Sequence[TranscriptSegment],
    ) -> List[Tuple[float, float, float, str]]:
        if not transcript_segments:
            return []

        texts = [segment.text for segment in transcript_segments if segment.text.strip()]
        if not texts:
            return []

        scores = self.ml_engine.compute_text_similarity(query, texts)
        if len(scores) == 0:
            return []

        scored_segments: List[Tuple[float, float, float, str]] = []
        text_index = 0
        for segment in transcript_segments:
            text = segment.text.strip()
            if not text:
                continue
            if text_index >= len(scores):
                break
            scored_segments.append(
                (
                    float(segment.start),
                    float(segment.end),
                    float(scores[text_index]),
                    text,
                )
            )
            text_index += 1
        return scored_segments

    @staticmethod
    def _interpolate_scores(
        source_ts: Tuple[float, ...],
        source_scores: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        if len(source_ts) == 0:
            return np.zeros(len(target_ts), dtype=np.float32)
        if len(source_ts) == 1:
            return np.full(len(target_ts), float(source_scores[0]), dtype=np.float32)
        source_ts_arr = np.asarray(source_ts, dtype=np.float64)
        source_scores_arr = np.asarray(source_scores, dtype=np.float32)
        return np.interp(
            target_ts,
            source_ts_arr,
            source_scores_arr,
            left=float(source_scores_arr[0]),
            right=float(source_scores_arr[-1]),
        ).astype(np.float32)

    def _fuse_visual_audio_timeline(
        self,
        visual_ts: Tuple[float, ...],
        visual_scores: np.ndarray,
        scored_audio_segments: Sequence[Tuple[float, float, float, str]],
    ) -> Tuple[Tuple[float, ...], np.ndarray, str]:
        if not scored_audio_segments:
            return visual_ts, visual_scores, "vision_only"

        timeline_points = set(float(t) for t in visual_ts)
        for start, end, _, _ in scored_audio_segments:
            timeline_points.add(float(start))
            timeline_points.add(float(end))
        fused_ts_arr = np.array(sorted(timeline_points), dtype=np.float64)

        visual_interp = self._interpolate_scores(visual_ts, visual_scores, fused_ts_arr)
        audio_scores = np.zeros(len(fused_ts_arr), dtype=np.float32)
        for index, timestamp in enumerate(fused_ts_arr):
            active_scores = [
                float(score)
                for start, end, score, _ in scored_audio_segments
                if float(start) <= float(timestamp) <= float(end)
            ]
            if active_scores:
                audio_scores[index] = max(active_scores)

        if not np.any(audio_scores > 0):
            return visual_ts, visual_scores, "vision_only"

        fused_scores = (
            visual_interp
            + self.AUDIO_BOOST_FACTOR * np.maximum(0.0, audio_scores - visual_interp)
        ).astype(np.float32)
        fused_scores = np.clip(fused_scores, 0.0, 1.0).astype(np.float32)
        return tuple(float(t) for t in fused_ts_arr), fused_scores, "vision_audio"

    def _interval_iou(
        self, left: tuple[float, float], right: tuple[float, float]
    ) -> float:
        start = max(left[0], right[0])
        end = min(left[1], right[1])
        intersection = max(0.0, end - start)
        union = max(left[1], right[1]) - min(left[0], right[0])
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _median_sample_spacing(timestamps: Tuple[float, ...]) -> float:
        if len(timestamps) < 2:
            return 0.25
        d = np.diff(np.asarray(timestamps, dtype=np.float64))
        d = d[d > 1e-6]
        if d.size == 0:
            return 0.25
        return float(np.median(d))

    def _refine_segment_times(
        self,
        timestamps: Tuple[float, ...],
        smoothed: np.ndarray,
        probs: np.ndarray,
        i_start: int,
        i_end: int,
    ) -> Tuple[float, float]:
        """
        Turn a coarse index interval into (start, end) seconds.
        Start is inclusive for recall; end is trimmed with a stricter threshold.
        """
        ts = np.asarray(timestamps, dtype=np.float64)
        n = len(ts)
        i0 = max(0, min(i_start, n - 1))
        i1 = max(i0, min(i_end, n - 1))

        seg_s = smoothed[i0 : i1 + 1]
        if seg_s.size == 0:
            return float(ts[i0]), float(ts[i1])

        peak_local = int(np.argmax(seg_s))
        i_peak = i0 + peak_local
        peak_v = float(smoothed[i_peak])
        p30 = float(np.percentile(seg_s, 30))
        p50 = float(np.percentile(seg_s, 50))

        tau_lead = max(p30, peak_v * 0.36)
        tau_trail = max(p50, peak_v * 0.62)

        k = i0
        while k < i_peak and smoothed[k] < tau_lead:
            k += 1

        j = i1
        while j > i_peak and smoothed[j] < tau_trail and probs[j] < tau_trail:
            j -= 1
        if j < k:
            j = k

        start_t = float(ts[k])
        end_t = float(ts[j])

        median_dt = self._median_sample_spacing(timestamps)
        min_span = max(1.0 * median_dt, 0.25)
        if end_t - start_t < min_span:
            end_t = min(float(ts[-1]), start_t + min_span)

        return start_t, end_t

    def _find_best_segment(
        self, probs: np.ndarray, timestamps: Tuple[float, ...], top_k: int = 5
    ) -> List[VideoSearchResult]:
        """
        Find top-k temporal segments from CLIP frame similarities.
        Boundaries follow score shape (variable length), not a fixed duration.
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

        # Stable hysteresis-based candidate generation.
        candidates: List[Tuple[int, int, float]] = []
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
                        conf = (
                            0.65 * float(np.max(segment_probs))
                            + 0.35 * float(np.mean(raw_probs))
                            if len(segment_probs) > 0
                            else 0.0
                        )
                        i0 = max(0, current_start_idx - 1)
                        i1 = min(len(timestamps) - 1, end_idx + 1)
                        candidates.append((i0, i1, conf))
                        current_start_idx = None

        if current_start_idx is not None:
            end_idx = len(probs) - 1
            segment_probs = smoothed[current_start_idx : end_idx + 1]
            raw_probs = probs[current_start_idx : end_idx + 1]
            conf = (
                0.65 * float(np.max(segment_probs)) + 0.35 * float(np.mean(raw_probs))
                if len(segment_probs) > 0
                else 0.0
            )
            i0 = max(0, current_start_idx - 1)
            i1 = min(len(timestamps) - 1, end_idx)
            candidates.append((i0, i1, conf))

        median_dt = self._median_sample_spacing(timestamps)

        if not candidates:
            ranked_frames = np.argsort(smoothed)[::-1][
                : max(1, min(top_k, len(smoothed)))
            ]
            span_idx = max(2, int(round(2.0 / max(median_dt, 0.05))))
            for idx in ranked_frames:
                idx = int(idx)
                i0 = max(0, idx - span_idx)
                i1 = min(len(timestamps) - 1, idx + span_idx)
                candidates.append((i0, i1, float(smoothed[idx])))

        candidates.sort(key=lambda x: x[2], reverse=True)

        results = []
        for i0, i1, score in candidates:
            start_t, end_t = self._refine_segment_times(
                timestamps, smoothed, probs, i0, i1
            )
            interval = (float(start_t), float(end_t))
            if any(
                self._interval_iou(interval, (res.start, res.end)) > 0.7
                for res in results
            ):
                continue
            results.append(
                VideoSearchResult(
                    start=start_t,
                    end=end_t,
                    confidence=float(max(0.0, min(1.0, score))),
                )
            )
            if len(results) >= top_k:
                break

        if not results:
            mid = float(timestamps[len(timestamps) // 2])
            span = max(4.0 * median_dt, 0.5)
            results.append(
                VideoSearchResult(
                    start=mid,
                    end=min(mid + span, float(timestamps[-1])),
                    confidence=0.1,
                )
            )

        return results
