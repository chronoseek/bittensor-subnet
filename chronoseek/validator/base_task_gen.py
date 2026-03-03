from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

class BaseTaskGenerator(ABC):
    """
    Abstract base class for Task Generators.
    This allows easy swapping between ActivityNet, Synthetic VLM, or other datasets.
    """
    
    @abstractmethod
    def generate_task(self) -> Tuple[str, str, Tuple[float, float]]:
        """
        Returns: (video_url, query, ground_truth_interval)
        """
        pass
