import cv2
from PIL import Image
from typing import List, Sequence, Tuple
import bittensor as bt


class FrameExtractor:
    """
    Handles video frame extraction.
    """

    @staticmethod
    def extract_frames(
        video_path: str, fps: int = 1
    ) -> List[Tuple[float, Image.Image]]:
        """
        Extract frames from a video at a specified FPS.
        Returns: List of (timestamp_sec, PIL.Image)
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                bt.logging.warning(f"Could not open video: {video_path}")
                return []

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                bt.logging.warning(f"Invalid FPS in video: {video_path}")
                return []

            frame_interval = int(video_fps / fps)
            if frame_interval == 0:
                frame_interval = 1

            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % frame_interval == 0:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.append((timestamp, pil_image))

                count += 1

            cap.release()
            return frames

        except Exception as e:
            bt.logging.error(f"Error extracting frames: {e}")
            return []

    @staticmethod
    def _merge_time_windows(
        windows: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Merge overlapping / adjacent intervals for fewer seeks and duplicate frames."""
        if not windows:
            return []
        sorted_w = sorted(
            ((float(a), float(b)) for a, b in windows),
            key=lambda x: x[0],
        )
        merged: List[List[float]] = [[sorted_w[0][0], sorted_w[0][1]]]
        for start, end in sorted_w[1:]:
            if start <= merged[-1][1] + 1e-3:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(float(a), float(b)) for a, b in merged]

    @staticmethod
    def extract_frames_in_windows(
        video_path: str,
        windows: Sequence[Tuple[float, float]],
        fps: float,
    ) -> List[Tuple[float, Image.Image]]:
        """
        Sample frames at approximately `fps` within each [start, end] window (seconds).
        Windows are merged before decoding to reduce duplicate work.
        """
        merged = FrameExtractor._merge_time_windows(windows)
        if not merged:
            return []

        frames: List[Tuple[float, Image.Image]] = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                bt.logging.warning(f"Could not open video: {video_path}")
                return []

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                bt.logging.warning(f"Invalid FPS in video: {video_path}")
                cap.release()
                return []

            min_interval_s = 1.0 / max(fps, 1e-6)

            for win_start, win_end in merged:
                if win_end <= win_start:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, win_start * 1000.0)
                last_sample_t = -1.0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if ts > win_end + 0.05:
                        break
                    if ts < win_start - 0.15:
                        continue
                    if (
                        last_sample_t < 0
                        or (ts - last_sample_t) >= min_interval_s - 1e-6
                    ):
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append((ts, Image.fromarray(rgb_frame)))
                        last_sample_t = ts

            cap.release()
            return frames
        except Exception as e:
            bt.logging.error(f"Error extracting frames in windows: {e}")
            return []
