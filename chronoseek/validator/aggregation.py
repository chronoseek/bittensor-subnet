from dataclasses import dataclass

from chronoseek.validator.telemetry import MinerTelemetrySummary


@dataclass(frozen=True)
class ScoreAggregationConfig:
    alpha: float = 0.1
    reliability_weight: float = 0.0
    consistency_weight: float = 0.0
    suspicion_weight: float = 0.0


@dataclass(frozen=True)
class MinerScoreComponents:
    quality_score: float
    reliability_score: float
    consistency_score: float
    suspicion_score: float
    final_score: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_multiplier(*, component_score: float, weight: float) -> float:
    weight = _clamp01(weight)
    return 1.0 - (weight * (1.0 - _clamp01(component_score)))


def update_miner_score(
    *,
    previous_score: float,
    instant_score: float,
    telemetry_summary: MinerTelemetrySummary | None,
    config: ScoreAggregationConfig = ScoreAggregationConfig(),
) -> MinerScoreComponents:
    alpha = _clamp01(config.alpha)
    quality_score = alpha * float(instant_score) + (1.0 - alpha) * float(previous_score)

    if telemetry_summary is None or telemetry_summary.attempts == 0:
        reliability_score = 1.0
        consistency_score = 1.0
        suspicion_score = 1.0
    else:
        reliability_score = 1.0 - telemetry_summary.error_rate
        consistency_score = 1.0 - min(1.0, telemetry_summary.score_stddev)
        flags = telemetry_summary.suspicion_flags()
        suspicion_score = max(0.0, 1.0 - (0.25 * len(flags)))

    multiplier = (
        _weighted_multiplier(
            component_score=reliability_score,
            weight=config.reliability_weight,
        )
        * _weighted_multiplier(
            component_score=consistency_score,
            weight=config.consistency_weight,
        )
        * _weighted_multiplier(
            component_score=suspicion_score,
            weight=config.suspicion_weight,
        )
    )
    final_score = max(0.0, quality_score * multiplier)
    return MinerScoreComponents(
        quality_score=quality_score,
        reliability_score=_clamp01(reliability_score),
        consistency_score=_clamp01(consistency_score),
        suspicion_score=_clamp01(suspicion_score),
        final_score=final_score,
    )
