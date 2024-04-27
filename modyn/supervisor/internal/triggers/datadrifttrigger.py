import logging
import os
import pathlib
from typing import Optional, Generator

# pylint: disable-next=no-name-in-module
from modyn.supervisor.internal.triggers.trigger import Trigger


logger = logging.getLogger(__name__)


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, trigger_config: dict):
        self.trigger_config = trigger_config
        self.pipeline_id: Optional[int] = None
        self.pipeline_config: Optional[dict] = None
        self.modyn_config: Optional[dict] = None
        self.base_dir: Optional[pathlib.Path] = None

        self.previous_trigger_id: Optional[int] = None
        self.previous_model_id: Optional[int] = None
        
        self.data_cache = []
        self.leftover_data_points = 0

        super().__init__(trigger_config)

    
    def _parse_trigger_config(self) -> None:
        self.detection_interval: int = 1000
        if "data_points_for_detection" in self.trigger_config.keys():
            self.detection_interval = self.trigger_config["data_points_for_detection"]
        assert self.detection_interval > 0, "data_points_for_trigger needs to be at least 1"
    
    def _create_dirs(self) -> None:
        assert self.pipeline_id is not None
        assert self.base_dir is not None

        self.exp_output_dir = self.base_dir / str(self.pipeline_id) / f"datadrift"
        self.drift_dir = self.exp_output_dir / "drift_results"
        os.makedirs(self.drift_dir, exist_ok=True)
        self.timers_dir = self.exp_output_dir / "timers"
        os.makedirs(self.timers_dir, exist_ok=True)

    def init_trigger(
        self, pipeline_id: int, pipeline_config: dict, modyn_config: dict, base_dir: pathlib.Path
    ) -> None:
        self.pipeline_id = pipeline_id
        self.pipeline_config = pipeline_config
        self.modyn_config = modyn_config
        self.base_dir = base_dir

        if "trigger_config" in self.pipeline_config["trigger"].keys():
            trigger_config = self.pipeline_config["trigger"]["trigger_config"]
            self._parse_trigger_config(trigger_config)
        self._create_dirs()


    def detect_drift(self, idx_start, idx_end) -> bool:
        pass


    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        """
        Use Generator here because this data drift trigger
        needs to wait for the previous trigger to finish and get the model
        """
        # add new data to data_cache
        self.data_cache.extend(new_data)

        unvisited_data_points = len(self.data_cache)
        untriggered_data_points = unvisited_data_points
        # the sliding window of data points for detection
        detection_idx_start = 0
        detection_idx_end = 0

        while unvisited_data_points >= self.detection_interval:
            unvisited_data_points -= self.detection_interval
            detection_idx_end += self.detection_interval
            if detection_idx_end <= self.leftover_data_points:
                continue

            # trigger_id doesn't always start from 0
            if self.previous_trigger_id is None:
                # if no previous trigger exists, always trigger retraining
                triggered = True
            else:
                # if exist previous trigger, detect drift
                triggered = self.detect_drift(detection_idx_start, detection_idx_end)


            if triggered:
                trigger_data_points = detection_idx_end - detection_idx_start
                trigger_idx = len(new_data) - (untriggered_data_points - trigger_data_points) - 1

                # update bookkeeping and sliding window
                untriggered_data_points -= trigger_data_points
                detection_idx_start = detection_idx_end
                yield trigger_idx

        # remove triggered data
        del self.data_cache[:detection_idx_start]
        self.leftover_data_points = detection_idx_end - detection_idx_start

    def inform_previous_trigger(self, previous_trigger_id: int) -> None:
        self.previous_trigger_id = previous_trigger_id

    def inform_previous_model(self, previous_model_id: int) -> None:
        self.previous_model_id = previous_model_id
