import os
import shutil
import subprocess
import tempfile

import cv2
from PIL import Image
from typing import List, Optional, Sequence, Tuple

from chronoseek.logging import logger


class FrameExtractor:
    """
    Handles video frame extraction.
    """

    @staticmethod
    def _transcode_to_h264(video_path: str) -> Optional[str]:
        """
        Re-encode `video_path` to h264 via the system ffmpeg binary.

        Used as a fallback when OpenCV can't decode a video: OpenCV's bundled
        FFmpeg build picks its native AV1 decoder, which on some
        builds/platforms only offers hardware pixel formats and fails
        outright ("Failed to get pixel format") instead of falling back to
        software decode - even though the system ffmpeg CLI decodes the same
        file fine via libdav1d. Returns the transcoded temp file path, or
        None if ffmpeg is unavailable or the transcode itself fails.
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg binary not found; cannot transcode unreadable video.")
            return None

        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-c:a",
                    "aac",
                    output_path,
                ],
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(f"Failed to run ffmpeg transcode for {video_path}: {exc}")
            FrameExtractor._remove_quietly(output_path)
            return None

        if (
            result.returncode != 0
            or not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0
        ):
            stderr_tail = result.stderr.decode("utf-8", errors="replace")[-500:]
            logger.warning(
                f"ffmpeg transcode failed for {video_path} "
                f"(exit code {result.returncode}): {stderr_tail}"
            )
            FrameExtractor._remove_quietly(output_path)
            return None

        return output_path

    @staticmethod
    def _remove_quietly(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _open_readable_capture(
        video_path: str,
    ) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
        """
        Open `video_path` for decoding, transcoding to h264 first if OpenCV
        can't actually read frames from it (see `_transcode_to_h264`).

        Returns (capture, transcoded_temp_path). `capture` is None if the
        video is unreadable even after a transcode attempt. The caller owns
        cleanup of transcoded_temp_path once done with the capture.
        """
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap, None
        cap.release()

        transcoded_path = FrameExtractor._transcode_to_h264(video_path)
        if transcoded_path is None:
            return None, None

        cap = cv2.VideoCapture(transcoded_path)
        if not cap.isOpened():
            cap.release()
            FrameExtractor._remove_quietly(transcoded_path)
            return None, None

        return cap, transcoded_path

    @staticmethod
    def extract_frames(
        video_path: str, fps: int = 1
    ) -> List[Tuple[float, Image.Image]]:
        """
        Extract frames from a video at a specified FPS.
        Returns: List of (timestamp_sec, PIL.Image)
        """
        frames = []
        transcoded_path = None
        try:
            cap, transcoded_path = FrameExtractor._open_readable_capture(video_path)

            if cap is None:
                logger.warning(f"Could not open video: {video_path}")
                return []

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                logger.warning(f"Invalid FPS in video: {video_path}")
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
            logger.error(f"Error extracting frames: {e}")
            return []
        finally:
            if transcoded_path:
                FrameExtractor._remove_quietly(transcoded_path)

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
        transcoded_path = None
        try:
            cap, transcoded_path = FrameExtractor._open_readable_capture(video_path)
            if cap is None:
                logger.warning(f"Could not open video: {video_path}")
                return []

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                logger.warning(f"Invalid FPS in video: {video_path}")
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
            logger.error(f"Error extracting frames in windows: {e}")
            return []
        finally:
            if transcoded_path:
                FrameExtractor._remove_quietly(transcoded_path)
