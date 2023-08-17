from datetime import datetime
from enum import Enum
import urllib.parse


LOG_STRING = "{timestamp} {event} {args}"

class Event(Enum):
    START_LOGGING = "start_logging"
    START_TRAINING = "start_training"
    START_EPOCH = "start_epoch"
    START_BATCH = "start_batch"
    PREPROCESS_END = "preprocess_end"
    BTS_START = "bts_start"
    BTS_COMPUTED_FORWARD = "bts_computed_forward"
    BTS_INFORMED_SAMPLES = "bts_informed_samples"
    BTS_SELECTED_SAMPLES = "bts_selected_samples"
    BTS_END = "bts_end"
    FORWARD_END = "forward_end"
    BACKWARD_END = "backward_end"
    STEP_END = "step_end"
    STB_START = "stb_start"
    STB_LABEL_COMPLETED = "stb_label_completed"
    STB_BATCH_START = "stb_batch_start"
    STB_PREPROCESS_END = "stb_preprocess_end"
    STB_FORWARD_END = "stb_forward_end"
    STB_INFORM_SAMPLES_END = "stb_inform_samples"
    STB_ALL_SAMPLES_INFORMED = "stb_all_samples_informed"
    STB_POINTS_SELECTED = "stb_points_selected"
    STB_POINTS_STORED = "stb_points_stored"
    STB_END = "stb_end"
    END_BATCH = "end_batch"
    END_EPOCH = "end_epoch"
    END_TRIGGER = "end_training"
    SELECTOR_START_TRIGGER = "selector_start_trigger"
    SELECTOR_START_PARTITION = "selector_start_partition"
    SELECTOR_STORED_SAMPLES = "selector_stored_samples"
    SELECTOR_END_STORING = "selector_end_storing"
    SELECTOR_END_TRIGGER = "selector_end_trigger"

class TimeLogger:
    def __init__(self):
        self.file = None

    def set_trigger(self, trigger):
        self.trigger = trigger
        if self.file is not None:
            self.file.close()
        self.file = open(f"./TIME_LOGS/timing_trigger_{trigger}.txt", "w")
        self.log(Event.START_LOGGING)

    def log(self, event: Event, args: str = "") -> None:
        assert self.file is not None
        if args != "":
            args = urllib.parse.quote_plus(args)
        self.file.write(
            LOG_STRING.format(timestamp=datetime.now().isoformat(), event=event, job_name=event.value,
                              args=args).strip() + "\n")
