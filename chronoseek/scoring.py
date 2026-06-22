import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from chronoseek.protocol_models import VideoSearchResult

GroundTruthInterval = Tuple[float, float]
GroundTruthIntervals = Iterable[GroundTruthInterval]
STRICT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class IntervalScoringConfig:
    """
    Controls the shape-aware interval quality score.

    IoU remains the primary signal. When penalties are enabled, the best IoU is
    multiplied by a shape factor made from center, boundary, and duration
    alignment. This keeps zero-overlap predictions at zero and prevents broad
    intervals from being rescued by auxiliary signals.
    """

    enabled: bool = True
    iou_weight: float = 0.75
    center_weight: float = 0.10
    boundary_weight: float = 0.10
    duration_weight: float = 0.05


DEFAULT_INTERVAL_SCORING = IntervalScoringConfig()
PURE_IOU_SCORING = IntervalScoringConfig(enabled=False)


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


def _positive_duration(start: float, end: float) -> float:
    return max(0.0, float(end) - float(start))


def _safe_alignment(error: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (abs(float(error)) / float(scale))))


def _duration_alignment(pred_len: float, gt_len: float) -> float:
    longer = max(float(pred_len), float(gt_len))
    if longer <= 0:
        return 0.0
    return max(0.0, min(1.0, min(float(pred_len), float(gt_len)) / longer))


def _normalized_weights(
    config: IntervalScoringConfig,
) -> tuple[float, float, float, float]:
    weights = (
        max(0.0, float(config.iou_weight)),
        max(0.0, float(config.center_weight)),
        max(0.0, float(config.boundary_weight)),
        max(0.0, float(config.duration_weight)),
    )
    total = sum(weights)
    if total <= 0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(weight / total for weight in weights)


def interval_quality_score(
    prediction: VideoSearchResult,
    ground_truth: GroundTruthInterval,
    *,
    config: IntervalScoringConfig = DEFAULT_INTERVAL_SCORING,
) -> float:
    """
    Score one prediction/ground-truth interval pair.

    The returned value is always in [0, 1]. With the default config it is IoU
    dampened by interval-shape alignment; with `enabled=False` it is pure IoU.
    """
    gt_start, gt_end = ground_truth
    iou = calculate_iou(prediction.start, prediction.end, gt_start, gt_end)
    if not config.enabled or iou <= 0.0:
        return iou

    pred_len = _positive_duration(prediction.start, prediction.end)
    gt_len = _positive_duration(gt_start, gt_end)
    if pred_len <= 0.0 or gt_len <= 0.0:
        return 0.0

    pred_center = (float(prediction.start) + float(prediction.end)) / 2.0
    gt_center = (float(gt_start) + float(gt_end)) / 2.0
    center_scale = max(pred_len, gt_len)
    center_alignment = _safe_alignment(pred_center - gt_center, center_scale)

    boundary_scale = gt_len
    start_alignment = _safe_alignment(
        float(prediction.start) - float(gt_start),
        boundary_scale,
    )
    end_alignment = _safe_alignment(
        float(prediction.end) - float(gt_end),
        boundary_scale,
    )
    boundary_alignment = (start_alignment + end_alignment) / 2.0

    duration_alignment = _duration_alignment(pred_len, gt_len)
    iou_weight, center_weight, boundary_weight, duration_weight = _normalized_weights(
        config
    )
    shape_multiplier = (
        iou_weight
        + center_weight * center_alignment
        + boundary_weight * boundary_alignment
        + duration_weight * duration_alignment
    )
    return max(0.0, min(1.0, iou * shape_multiplier))


def best_interval_quality(
    predictions: List[VideoSearchResult],
    ground_truths: GroundTruthIntervals,
    *,
    config: IntervalScoringConfig = DEFAULT_INTERVAL_SCORING,
) -> float:
    max_score = 0.0
    normalized_ground_truths = list(ground_truths)
    if not predictions or not normalized_ground_truths:
        return 0.0

    for pred in predictions:
        for ground_truth in normalized_ground_truths:
            score = interval_quality_score(pred, ground_truth, config=config)
            if score > max_score:
                max_score = score

    return max_score


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
    latency: float,  # Kept for API compatibility; latency is handled by validators.
    *,
    clip_duration: float | None = None,
    max_prediction_duration_seconds: float | None = None,
    score_top_k: int | None = None,
    interval_scoring_config: IntervalScoringConfig = DEFAULT_INTERVAL_SCORING,
) -> float:
    """
    Score a miner's response using the best interval quality across predictions and ground truths.
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

    return best_interval_quality(
        filtered_predictions,
        ground_truths,
        config=interval_scoring_config,
    )


def passes_strict_iou(score: float, threshold: float = STRICT_IOU_THRESHOLD) -> bool:
    return score >= threshold
