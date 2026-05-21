import math
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


def _ground_truth_list(
    ground_truth: GroundTruthInterval | GroundTruthIntervals,
) -> list[GroundTruthInterval]:
    if (
        isinstance(ground_truth, tuple)
        and len(ground_truth) == 2
        and all(isinstance(value, (int, float)) for value in ground_truth)
    ):
        return [(float(ground_truth[0]), float(ground_truth[1]))]
    return [(float(start), float(end)) for start, end in ground_truth]


def _valid_prediction(
    prediction: VideoSearchResult,
    *,
    clip_duration: float | None,
    max_prediction_duration_seconds: float | None,
    ground_truths: list[GroundTruthInterval],
) -> bool:
    start = float(prediction.start)
    end = float(prediction.end)
    if not (math.isfinite(start) and math.isfinite(end)):
        return False
    if start < 0 or end <= start:
        return False
    if clip_duration is not None and end > float(clip_duration):
        return False

    duration = end - start
    if max_prediction_duration_seconds is None:
        return True

    max_duration = float(max_prediction_duration_seconds)
    if max_duration <= 0 or duration <= max_duration:
        return True

    return any((gt_end - gt_start) > max_duration for gt_start, gt_end in ground_truths)


def valid_predictions(
    predictions: List[VideoSearchResult],
    ground_truths: list[GroundTruthInterval],
    *,
    clip_duration: float | None = None,
    max_prediction_duration_seconds: float | None = None,
    score_top_k: int | None = None,
) -> list[VideoSearchResult]:
    valid: list[VideoSearchResult] = []
    seen: set[tuple[float, float]] = set()
    candidate_predictions = predictions
    if score_top_k is not None:
        candidate_predictions = predictions[: max(0, int(score_top_k))]

    for prediction in candidate_predictions:
        if not _valid_prediction(
            prediction,
            clip_duration=clip_duration,
            max_prediction_duration_seconds=max_prediction_duration_seconds,
            ground_truths=ground_truths,
        ):
            continue
        key = (round(float(prediction.start), 3), round(float(prediction.end), 3))
        if key in seen:
            continue
        seen.add(key)
        valid.append(prediction)

    return valid


def score_response(
    predictions: List[VideoSearchResult],
    ground_truth: GroundTruthInterval | GroundTruthIntervals,
    latency: float,  # Kept for API compatibility; current scoring is IoU-only.
    *,
    clip_duration: float | None = None,
    max_prediction_duration_seconds: float | None = None,
    score_top_k: int | None = None,
) -> float:
    """
    Score a miner's response using the best IoU across predictions and ground truths.
    This returns a continuous value in [0, 1].
    """
    if not predictions:
        return 0.0

    ground_truths = _ground_truth_list(ground_truth)
    filtered_predictions = valid_predictions(
        predictions,
        ground_truths,
        clip_duration=clip_duration,
        max_prediction_duration_seconds=max_prediction_duration_seconds,
        score_top_k=score_top_k,
    )
    if not filtered_predictions:
        return 0.0

    return best_iou(filtered_predictions, ground_truths)


def passes_strict_iou(score: float, threshold: float = STRICT_IOU_THRESHOLD) -> bool:
    return score >= threshold
