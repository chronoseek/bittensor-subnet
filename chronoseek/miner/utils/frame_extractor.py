import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
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
