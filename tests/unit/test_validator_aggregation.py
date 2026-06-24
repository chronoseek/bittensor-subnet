import pytest

from chronoseek.validator.aggregation import (
    ScoreAggregationConfig,
    update_miner_score,
)
from chronoseek.validator.telemetry import MinerTelemetryEvent, ValidatorTelemetryState


def test_score_aggregation_default_reproduces_quality_ema():
    components = update_miner_score(
        previous_score=0.4,
        instant_score=1.0,
        telemetry_summary=None,
        config=ScoreAggregationConfig(alpha=0.1),
    )

    assert components.quality_score == pytest.approx(0.46)
    assert components.final_score == pytest.approx(0.46)
    assert components.reliability_score == 1.0
    assert components.consistency_score == 1.0
    assert components.suspicion_score == 1.0


def test_score_aggregation_can_apply_reliability_penalty():
    telemetry = ValidatorTelemetryState()
    for _ in range(5):
        telemetry.record(
            MinerTelemetryEvent(
                uid=1,
                score=0.0,
                latency=10.0,
                task_family="hardened-activitynet",
                failure_kind="timeout",
            )
        )

    components = update_miner_score(
        previous_score=0.5,
        instant_score=1.0,
        telemetry_summary=telemetry.summaries[1],
        config=ScoreAggregationConfig(
            alpha=0.1,
            reliability_weight=0.5,
            suspicion_weight=0.5,
        ),
    )

    assert components.quality_score == pytest.approx(0.55)
    assert components.reliability_score == 0.0
    assert components.suspicion_score < 1.0
    assert components.final_score < components.quality_score


def test_score_aggregation_keeps_penalties_opt_in():
    telemetry = ValidatorTelemetryState()
    for _ in range(5):
        telemetry.record(
            MinerTelemetryEvent(
                uid=1,
                score=0.0,
                latency=10.0,
                task_family="hardened-activitynet",
                failure_kind="timeout",
            )
        )

    components = update_miner_score(
        previous_score=0.5,
        instant_score=1.0,
        telemetry_summary=telemetry.summaries[1],
        config=ScoreAggregationConfig(alpha=0.1),
    )

    assert components.final_score == components.quality_score
