import math
from typing import List, Tuple
from chronoseek.schemas import VideoSearchResult


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


def score_response(
    predictions: List[VideoSearchResult],
    ground_truth: Tuple[float, float],
    latency: float,  # Kept for API compatibility, ignored in MVP
) -> float:
    """
    Score a miner's response based on MVP Strict IoU rules.

    Rule 1: Binary Pass/Fail
    - If max(IoU) >= 0.5: Score = 1.0
    - If max(IoU) < 0.5: Score = 0.0

    Rule 2: Oversized Interval Penalty (Optional for MVP, but good practice)
    - If duration(pred) > 2 * duration(gt), apply penalty?
    - MVP Spec says "Binary Pass/Fail scoring... removes all subjectivity".
    - So we stick to pure binary for now.
    """
    if not predictions:
        return 0.0

    gt_start, gt_end = ground_truth

    # Find best matching prediction (max IoU)
    max_iou = 0.0
    for pred in predictions:
        iou = calculate_iou(pred.start, pred.end, gt_start, gt_end)
        if iou > max_iou:
            max_iou = iou

    # MVP Strict Threshold
    if max_iou >= 0.5:
        return 1.0
    else:
        return 0.0
