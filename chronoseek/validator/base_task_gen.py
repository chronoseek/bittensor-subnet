from abc import ABC, abstractmethod
from typing import List, Tuple

GroundTruthInterval = Tuple[float, float]
GroundTruthIntervals = List[GroundTruthInterval]


class BaseTaskGenerator(ABC):
    """
    Abstract base class for Task Generators.
    This allows easy swapping between ActivityNet, Synthetic VLM, or other datasets.
    """

    @abstractmethod
    def generate_task(self) -> Tuple[str, str, GroundTruthIntervals]:
        """
        Returns: (video_url, query, ground_truth_intervals)
        """
        pass
