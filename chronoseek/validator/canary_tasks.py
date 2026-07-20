from dataclasses import replace

from chronoseek.validator.task_models import ValidationTask


class CanaryTaskPolicy:
    """Converts a configured fraction of normal tasks into validator canaries."""

    def __init__(self, *, rate: float, absent_queries: tuple[str, ...]):
        self.rate = max(0.0, min(1.0, float(rate)))
        self.absent_queries = tuple(query.strip() for query in absent_queries if query.strip())

    def apply(self, *, task: ValidationTask, rng) -> ValidationTask:
        if self.rate <= 0.0 or rng.random() >= self.rate:
            return task

        candidates = ["absent"]
        if task.hard_negative_count > 0:
            candidates.append("hard-negative")
        if len(task.ground_truths) > 1:
            candidates.append("repeated")

        canary_kind = rng.choice(candidates)
        metadata = {
            **task.transform_metadata,
            "canary_kind": canary_kind,
            "canary_source": "validator",
        }
        if canary_kind == "absent":
            query = rng.choice(
                self.absent_queries or ("an event that is not present",)
            )
            return replace(
                task,
                task_family="canary-absent",
                query=query,
                ground_truths=[],
                canary_kind=canary_kind,
                expects_empty_response=True,
                transform_metadata=metadata,
            )

        return replace(
            task,
            task_family=f"canary-{canary_kind}",
            canary_kind=canary_kind,
            expects_empty_response=False,
            transform_metadata=metadata,
        )
