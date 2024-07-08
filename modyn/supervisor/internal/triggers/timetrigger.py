from __future__ import annotations

from typing import Generator

from modyn.config.schema.pipeline import TimeTriggerConfig
from modyn.supervisor.internal.triggers.models import TriggerPolicyEvaluationLog
from modyn.supervisor.internal.triggers.trigger import Trigger


class TimeTrigger(Trigger):
    """Triggers after a certain amount of time has passed.
    Uses the sample timestamps and not time at supervisor to measure passed time.
    Clock starts with the first observed datapoint"""

    def __init__(self, config: TimeTriggerConfig):
        self.config = config
        self.next_trigger_at: int | None = None

        if self.config.every_seconds < 1:
            raise ValueError(f"trigger_every must be > 0, but is {self.config.every_seconds}")

        super().__init__()

    def inform(
        self, new_data: list[tuple[int, int, int]], log: TriggerPolicyEvaluationLog | None = None
    ) -> Generator[int, None, None]:
        if self.next_trigger_at is None:
            yield -1 # We want an empty trigger at the beginning such that we have a random model to start with
            
            if self.config.start_timestamp is not None:
                self.next_trigger_at = self.config.start_timestamp + self.config.every_seconds
            else:
                if len(new_data) > 0:
                    self.next_trigger_at = new_data[0][1] + self.config.every_seconds  # new_data is sorted
                else:
                    return

        max_timestamp = new_data[-1][1]  # new_data is sorted
        triggering_indices = []

        while self.next_trigger_at <= max_timestamp:
            # The next line gets the first item which has a timestamp larger or equal to the triggering timestamp
            try:
                idx = next(idx for (idx, (_, timestamp, _)) in enumerate(new_data) if timestamp >= self.next_trigger_at)
            except StopIteration:
                break
            # This index `idx` describes the first item not belonging to the trigger.
            # Hence, the previous item causes a trigger.
            # If this is the first item, then we need to emit a trigger for index -1.
            # This means that there was a trigger before the first item that we got informed about
            # However, there might have been multiple triggers, e.g., if there is one trigger every second
            # and 5 seconds have passed since the last item came through
            # This is caught by our while loop which increases step by step for `config.every_seconds`.

            triggering_indices.append(idx - 1)
            self.next_trigger_at += self.config.every_seconds

        yield from triggering_indices
