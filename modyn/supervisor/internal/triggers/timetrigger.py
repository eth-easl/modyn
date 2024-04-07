from collections.abc import Generator
from typing import Optional

from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.utils import convert_timestr_to_seconds, validate_timestr


class TimeTrigger(Trigger):
    """Triggers after a certain amount of time has passed.
    Uses the sample timestamps and not time at supervisor to measure passed time.
    Clock starts with the first observed datapoint"""

    def __init__(self, trigger_config: dict):
        if "trigger_every" not in trigger_config.keys():
            raise ValueError("Trigger config is missing `trigger_every` field")

        timestr = trigger_config["trigger_every"]
        if not validate_timestr(timestr):
            raise ValueError(f"Invalid time string: {timestr}\nValid format is <number>[s|m|h|d|w].")

        self.trigger_every_s: int = convert_timestr_to_seconds(trigger_config["trigger_every"])
        self.next_trigger_at: Optional[int] = None

        if self.trigger_every_s < 1:
            raise ValueError(f"trigger_every must be > 0, but is {self.trigger_every_s}")

        super().__init__(trigger_config)

    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        if self.next_trigger_at is None:
            if len(new_data) > 0:
                self.next_trigger_at = new_data[0][1] + self.trigger_every_s - 1  # new_data is sorted
            else:
                return

        max_timestamp = new_data[-1][1]  # new_data is sorted
        triggering_indices = []

        while self.next_trigger_at <= max_timestamp:
            print(f">>>>>>>>>>>>>>next trigger {self.next_trigger_at}")
            
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
            # This is caught by our while loop which increases step by step for `trigger_every_s`.

            triggering_indices.append(idx - 1)
            self.next_trigger_at += self.trigger_every_s

        yield from triggering_indices
