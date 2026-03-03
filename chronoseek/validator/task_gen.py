import random
import json
import os
from typing import Tuple, List, Dict
from chronoseek.validator.base_task_gen import BaseTaskGenerator


class ActivityNetTaskGenerator(BaseTaskGenerator):
    """
    Generates tasks based on the ActivityNet Captions dataset (MVP Scope).
    """

    def __init__(self, dataset_path: str = "activity_net.v1-3.min.json"):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """
        Load the ActivityNet dataset.
        For MVP, if the file doesn't exist, we fall back to a hardcoded mini-set
        that mimics the structure of ActivityNet.
        """
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r") as f:
                data = json.load(f)
                # Parse ActivityNet format into a list of tasks
                tasks = []
                for vid_id, content in data.get("database", {}).items():
                    url = content.get("url", "")
                    if not url:
                        continue

                    for i, sentence in enumerate(content.get("sentences", [])):
                        timestamp = content["timestamps"][i]  # [start, end]
                        tasks.append(
                            {
                                "video_url": url,
                                "query": sentence,
                                "ground_truth": (
                                    float(timestamp[0]),
                                    float(timestamp[1]),
                                ),
                            }
                        )
                return tasks
        else:
            # Fallback MVP Dataset (Public Domain / CC videos)
            # Mimicking ActivityNet structure
            return [
                {
                    "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
                    "query": "the big rabbit wakes up from his hole",
                    "ground_truth": (10.0, 25.0),
                },
                {
                    "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
                    "query": "two characters are arguing on the bridge",
                    "ground_truth": (50.0, 70.0),
                },
                {
                    "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
                    "query": "a robot runs down the street",
                    "ground_truth": (120.0, 135.0),
                },
            ]

    def generate_task(self) -> Tuple[str, str, Tuple[float, float]]:
        """
        Returns: (video_url, query, ground_truth_interval)
        """
        task = random.choice(self.dataset)
        return task["video_url"], task["query"], task["ground_truth"]
