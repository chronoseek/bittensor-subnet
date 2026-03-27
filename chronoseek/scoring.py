from typing import Iterable, List, Tuple
from chronoseek.protocol_models import VideoSearchResult

GroundTruthInterval = Tuple[float, float]
GroundTruthIntervals = Iterable[GroundTruthInterval]
STRICT_IOU_THRESHOLD = 0.5


def calculate_iou(
    pred_start: float, pred_end: float, gt_start: float, gt_end: float
) -> float:
    """
    Calculate Intersection over Union (IoU) between two time intervals.
    """
    # Calculate intersection
    start = max(pred_start, gt_start)
    end = min(pred_end, gt_end)
    intersection = max(0.0, end - start)

    # Calculate union
    pred_len = pred_end - pred_start
    gt_len = gt_end - gt_start

    if pred_len <= 0 or gt_len <= 0:
        return 0.0

    union = pred_len + gt_len - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def best_iou(
    predictions: List[VideoSearchResult],
    ground_truths: GroundTruthIntervals,
) -> float:
    """
    Return the best IoU across all prediction and ground-truth interval pairs.
    """
    max_iou = 0.0
    normalized_ground_truths = list(ground_truths)
    if not predictions or not normalized_ground_truths:
        return 0.0

    for pred in predictions:
        for gt_start, gt_end in normalized_ground_truths:
            iou = calculate_iou(pred.start, pred.end, gt_start, gt_end)
            if iou > max_iou:
                max_iou = iou

    return max_iou


def score_response(
    predictions: List[VideoSearchResult],
    ground_truth: GroundTruthInterval | GroundTruthIntervals,
    latency: float,  # Kept for API compatibility, ignored in MVP
) -> float:
    """
    Score a miner's response using the best IoU across predictions and ground truths.
    This returns a continuous value in [0, 1].
    """
    if not predictions:
        return 0.0

    if (
        isinstance(ground_truth, tuple)
        and len(ground_truth) == 2
        and all(isinstance(value, (int, float)) for value in ground_truth)
    ):
        ground_truths = [ground_truth]
    else:
        ground_truths = list(ground_truth)

    return best_iou(predictions, ground_truths)


def passes_strict_iou(score: float, threshold: float = STRICT_IOU_THRESHOLD) -> bool:
    return score >= threshold
