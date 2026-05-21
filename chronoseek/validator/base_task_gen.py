from abc import ABC, abstractmethod
from typing import List, Tuple

GroundTruthInterval = Tuple[float, float]
GroundTruthIntervals = List[GroundTruthInterval]
LegacyTask = Tuple[str, str, GroundTruthIntervals]


class BaseTaskGenerator(ABC):
    """
    Abstract base class for Task Generators.
    This allows easy swapping between ActivityNet, Synthetic VLM, or other datasets.
    """

    @abstractmethod
    def generate_task(self):
        """
        Returns either:
        - legacy tuple: (video_url, query, ground_truth_intervals)
        - hardened ValidationTask object
        """
        pass
