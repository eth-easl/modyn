from modyn.backend.supervisor.internal.trigger import Trigger
from modyn.utils import validate_timestr, convert_timestr_to_seconds
from typing import Optional

class TimeTrigger(Trigger):
    """Triggers after a certain amount of time has passed.
    Uses the sample timestamps and not time at supervisor to measure passed time.
    Clock starts with the first observed datapoint"""

    def __init__(self, trigger_config: dict):
        if "trigger_every" not in trigger_config.keys():
            raise ValueError("Trigger config is missing `trigger_every` field") # TODO add to schema

        timestr = trigger_config["trigger_every"]
        if not validate_timestr(timestr):
            raise ValueError(f"Invalid time string: {timestr}\nValid format is <number>[s|m|h|d|w].")

        self.trigger_every_ms: int = convert_timestr_to_seconds(trigger_config["trigger_every"]) * 1000
        self.next_trigger_at: Optional[int] = None

        if self.trigger_every_ms < 1:
            raise ValueError(f"trigger_every must be > 0, but is {self.trigger_every_ms}")

        super().__init__(trigger_config)

    def inform(self, new_data: list[tuple[str, int, int]]) -> list[int]:
        if self.next_trigger_at is None:
            if len(new_data) > 0:
                self.next_trigger_at = min([timestamp for _, timestamp, _ in new_data]) + self.trigger_every_ms
            else:
                return []

        # We use the fact that new_data is sorted
        max_timestamp = new_data[-1][1]
        triggering_indices = []
        search_start = 0

        while self.next_trigger_at <= max_timestamp:
            # The next line gets the first item which has a timestamp larger or equal to the triggering timestamp
            idx = next(idx for (idx, (_, timestamp, _)) in list(enumerate(new_data))[search_start:] if timestamp >= self.next_trigger_at)
            # Since this is the first item not belonging to the trigger, we need to fetch the index of the previous item
            # TODO: We might need to emit multiple -1 in case idx == 0, in case we trigger every s and 5s have passed then 5 empty triggers

            triggering_indices.append(idx - 1)
            self.next_trigger_at += self.trigger_every_ms
            search_start = idx

            # TODO write tests for selector/supervisor that we can deal with -1 indices (also multiple)

        return triggering_indices
