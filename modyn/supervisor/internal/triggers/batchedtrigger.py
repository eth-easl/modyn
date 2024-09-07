from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Generator

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.common.batched import BatchedTriggerConfig
from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.utils.models import TriggerPolicyEvaluationLog
from modyn.supervisor.internal.triggers.utils.warmuptrigger import WarmupTrigger

logger = logging.getLogger(__name__)


class BatchedTrigger(Trigger):
    """Abstract child of Trigger that implements triggering in discrete
    intervals."""

    def __init__(self, config: BatchedTriggerConfig) -> None:
        self.config = config

        # allows to detect drift in a fixed interval
        self._sample_left_until_detection = config.evaluation_interval_data_points
        self._last_detection_interval: list[tuple[int, int]] = []

        self._leftover_data: list[tuple[int, int]] = []
        """Stores data that was not processed in the last inform call because
        the detection interval was not filled."""

        self.warmup_trigger = WarmupTrigger(
            warmup_intervals=config.warmup_intervals, warmup_policy=config.warmup_policy
        )
        """Warmup decisions have to be delegated to the warmup trigger by the
        subclass of BatchedTrigger."""

    @override
    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        new_key_ts = self._leftover_data + [(key, timestamp) for key, timestamp, _ in new_data]
        # reappending the leftover data to the new data requires incrementing the sample left until detection
        self._sample_left_until_detection += len(self._leftover_data)

        # index of the first unprocessed data point in the batch
        processing_head_in_batch = 0

        # Go through remaining data in new data in batches of `detect_interval`
        while True:
            if self._sample_left_until_detection - len(new_key_ts) > 0:
                # No detection in this trigger because of too few data points to fill detection interval
                self._leftover_data = new_key_ts
                self._sample_left_until_detection -= len(new_key_ts)
                return

            # At least one detection, fill up window up to that detection
            next_detection_interval = new_key_ts[: self._sample_left_until_detection]

            # Update the remaining data
            processing_head_in_batch += len(next_detection_interval)
            new_key_ts = new_key_ts[len(next_detection_interval) :]

            # we need to return an index in the `new_data`. Therefore, we need to subtract number of samples in the
            # leftover data from the processing head in batch; -1 is required as the head points to the first
            # unprocessed data point
            trigger_candidate_idx = min(
                max(processing_head_in_batch - len(self._leftover_data) - 1, 0),
                len(new_data) - 1,
            )

            # Reset for next detection
            self._sample_left_until_detection = self.config.evaluation_interval_data_points

            # ----------------------------------------------- Detection ---------------------------------------------- #

            triggered = self._evaluate_batch(next_detection_interval, trigger_candidate_idx, log=log)
            self._last_detection_interval = next_detection_interval

            # ----------------------------------------------- Response ----------------------------------------------- #

            if triggered:
                yield trigger_candidate_idx

    @abstractmethod
    def _evaluate_batch(
        self,
        batch: list[tuple[int, int]],
        trigger_candidate_idx: int,
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> bool: ...
