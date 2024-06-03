from typing import Literal

from modyn.config.schema.pipeline.evaluation import StaticEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalTrigger


class StaticEvalTrigger(EvalTrigger):

    def __init__(self, config: StaticEvalTriggerConfig) -> None:
        super().__init__()
        self.unit: Literal["epoch", "sample_index"] = config.unit
        self.remaining_points_in_time = config.at
        self.sample_counter = 0

        if self.unit == "epoch":
            self.evaluation_backlog = [
                EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=epoch) for epoch in sorted(config.at)
            ]

    def inform(self, new_data: list[tuple[int, int, int]]) -> None:
        """Inform the trigger about a batch of new data.

        StaticEvalTrigger will trigger on the first sample after the specified point in time.

        Side Effects:
            The trigger appends EvalRequests internally for  all points in time where evaluation is required.
        """
        if len(new_data) == 0 or self.unit == "epoch":
            return

        if self.unit == "sample_index":
            ready_trigger_sample_indexes = [
                trigger_sample_idx
                for trigger_sample_idx in self.remaining_points_in_time
                if trigger_sample_idx <= self.sample_counter + len(new_data) - 1  # rhs: index of last sample in batch
            ]

            for trigger_sample_idx in ready_trigger_sample_indexes:
                trigger_idx_in_batch = trigger_sample_idx - self.sample_counter

                if (
                    len(self.evaluation_backlog) > 0
                    and self.evaluation_backlog[-1].sample_timestamp == new_data[trigger_idx_in_batch][1]
                ):
                    continue

                self.evaluation_backlog.append(
                    EvalCandidate(
                        sample_index=trigger_sample_idx,
                        sample_id=new_data[trigger_idx_in_batch][0],
                        sample_timestamp=new_data[trigger_idx_in_batch][1],
                    )
                )

            for trigger_sample_idx in ready_trigger_sample_indexes:
                self.remaining_points_in_time.remove(trigger_sample_idx)

        elif self.unit == "epoch":
            pass  # already pre-determined with config

        else:
            raise ValueError(f"Invalid unit: {self.unit}")

        # Update trackers
        self.sample_counter += len(new_data)
