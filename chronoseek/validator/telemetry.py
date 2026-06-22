import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MinerTelemetryEvent:
    uid: int
    score: float
    latency: float
    task_family: str
    canary_kind: str | None = None
    transform_id: str | None = None
    hard_negative_count: int = 0
    failure_kind: str | None = None
    protocol_code: str | None = None
    status_code: int | None = None
    top_start: float | None = None
    top_end: float | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def failed(self) -> bool:
        return bool(self.failure_kind)

    @property
    def prediction_duration(self) -> float | None:
        if self.top_start is None or self.top_end is None:
            return None
        return max(0.0, float(self.top_end) - float(self.top_start))


@dataclass
class MinerTelemetrySummary:
    uid: int
    attempts: int = 0
    failures: int = 0
    timeouts: int = 0
    total_score: float = 0.0
    total_latency: float = 0.0
    canary_attempts: int = 0
    canary_score: float = 0.0
    hard_negative_attempts: int = 0
    duration_buckets: dict[str, int] = field(default_factory=dict)

    def record(self, event: MinerTelemetryEvent) -> None:
        self.attempts += 1
        self.total_score += float(event.score)
        self.total_latency += max(0.0, float(event.latency))
        if event.failed:
            self.failures += 1
        if event.failure_kind == "timeout":
            self.timeouts += 1
        if event.canary_kind:
            self.canary_attempts += 1
            self.canary_score += float(event.score)
        if int(event.hard_negative_count) > 0:
            self.hard_negative_attempts += 1

        duration = event.prediction_duration
        if duration is not None:
            bucket = f"{round(duration, 1):.1f}"
            self.duration_buckets[bucket] = self.duration_buckets.get(bucket, 0) + 1

    @property
    def average_score(self) -> float:
        return self.total_score / self.attempts if self.attempts else 0.0

    @property
    def average_latency(self) -> float:
        return self.total_latency / self.attempts if self.attempts else 0.0

    @property
    def error_rate(self) -> float:
        return self.failures / self.attempts if self.attempts else 0.0

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.attempts if self.attempts else 0.0

    @property
    def average_canary_score(self) -> float:
        return self.canary_score / self.canary_attempts if self.canary_attempts else 0.0

    @property
    def dominant_duration_rate(self) -> float:
        if not self.duration_buckets:
            return 0.0
        return max(self.duration_buckets.values()) / sum(self.duration_buckets.values())

    def suspicion_flags(self) -> list[str]:
        flags: list[str] = []
        if self.attempts >= 5 and self.error_rate >= 0.50:
            flags.append("high_error_rate")
        if self.attempts >= 5 and self.timeout_rate >= 0.25:
            flags.append("high_timeout_rate")
        if sum(self.duration_buckets.values()) >= 5 and self.dominant_duration_rate >= 0.80:
            flags.append("repeated_prediction_duration")
        if (
            self.canary_attempts >= 3
            and self.average_score >= 0.50
            and self.average_canary_score <= 0.10
        ):
            flags.append("canary_underperformance")
        return flags

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "attempts": self.attempts,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "average_score": self.average_score,
            "average_latency": self.average_latency,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
            "canary_attempts": self.canary_attempts,
            "average_canary_score": self.average_canary_score,
            "hard_negative_attempts": self.hard_negative_attempts,
            "dominant_duration_rate": self.dominant_duration_rate,
            "duration_buckets": dict(sorted(self.duration_buckets.items())),
            "suspicion_flags": self.suspicion_flags(),
        }


class ValidatorTelemetryState:
    def __init__(self, *, max_events: int = 1000):
        self.max_events = max(1, int(max_events))
        self.events: list[MinerTelemetryEvent] = []
        self.summaries: dict[int, MinerTelemetrySummary] = {}

    def record(self, event: MinerTelemetryEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

        uid = int(event.uid)
        summary = self.summaries.setdefault(uid, MinerTelemetrySummary(uid=uid))
        summary.record(event)

    def snapshot(self) -> dict:
        return {
            "generated_at": time.time(),
            "events": [asdict(event) for event in self.events],
            "summaries": {
                str(uid): summary.to_dict()
                for uid, summary in sorted(self.summaries.items())
            },
        }

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True))
