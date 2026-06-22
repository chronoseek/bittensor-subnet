import json

from chronoseek.validator.telemetry import (
    MinerTelemetryEvent,
    ValidatorTelemetryState,
)


def test_validator_telemetry_summarizes_scores_errors_and_latency(tmp_path):
    telemetry = ValidatorTelemetryState(max_events=3)
    for _ in range(5):
        telemetry.record(
            MinerTelemetryEvent(
                uid=7,
                score=0.0,
                latency=30.0,
                task_family="hardened-activitynet",
                failure_kind="timeout",
            )
        )

    summary = telemetry.summaries[7]

    assert len(telemetry.events) == 3
    assert summary.attempts == 5
    assert summary.failures == 5
    assert summary.timeouts == 5
    assert summary.error_rate == 1.0
    assert summary.timeout_rate == 1.0
    assert "high_error_rate" in summary.suspicion_flags()
    assert "high_timeout_rate" in summary.suspicion_flags()

    output_path = tmp_path / "telemetry.json"
    telemetry.save_json(output_path)
    payload = json.loads(output_path.read_text())
    assert payload["summaries"]["7"]["attempts"] == 5


def test_validator_telemetry_flags_repeated_prediction_durations():
    telemetry = ValidatorTelemetryState()
    for _ in range(5):
        telemetry.record(
            MinerTelemetryEvent(
                uid=3,
                score=0.2,
                latency=1.0,
                task_family="hardened-activitynet",
                top_start=10.0,
                top_end=20.0,
            )
        )

    summary = telemetry.summaries[3]

    assert summary.dominant_duration_rate == 1.0
    assert "repeated_prediction_duration" in summary.suspicion_flags()
