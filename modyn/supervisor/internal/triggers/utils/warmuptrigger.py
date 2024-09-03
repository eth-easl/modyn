from modyn.config.schema.pipeline.trigger.simple import SimpleTriggerConfig
from modyn.supervisor.internal.triggers.utils.factory import instantiate_trigger


class WarmupTrigger:
    """Utility class that wraps a SimpleTrigger which can be used in
    BatchedTriggers as a warmup trigger."""

    def __init__(
        self,
        warmup_intervals: int | None = None,
        warmup_policy: SimpleTriggerConfig | None = None,
    ):
        # [WARMUP CONFIGURATION]
        self._delegation_counter = 0
        self._warmup_intervals = (warmup_intervals if warmup_policy else 0) or 0

        # warmup policy (used as drop in replacement for the yet uncalibrated drift policy)
        self.trigger = instantiate_trigger(warmup_policy.id, warmup_policy) if warmup_policy else None

    @property
    def completed(self) -> bool:
        return self._delegation_counter >= self._warmup_intervals

    def delegate_inform(self, batch: list[tuple[int, int]]) -> bool:
        """Consult the warmup trigger with the given batch."""
        if self.trigger is None:
            return True
        assert not self.completed
        self._delegation_counter += 1

        delegated_trigger_results = (
            list(
                self.trigger.inform([(idx, time, 0) for (idx, time) in batch]),
            )
            if self.trigger
            else None
        )

        return len(delegated_trigger_results or []) > 0
