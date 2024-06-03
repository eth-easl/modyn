from modyn.config.schema.pipeline.evaluation import PeriodicEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalTrigger


class PeriodicEvalTrigger(EvalTrigger):

    def __init__(self, config: PeriodicEvalTriggerConfig) -> None:
        super().__init__()
        self.unit_type = config.unit_type
        self.every = config.every_sec_or_samples
        self.sample_counter = 0
        self.remaining_samples_for_next_trigger = 0

        if self.unit_type == "epoch":
            assert config.start_timestamp is not None
            assert config.end_timestamp is not None
            self.evaluation_backlog = [
                EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=epoch)
                for epoch in range(config.start_timestamp, config.end_timestamp + 1, self.every)
            ]

    def inform(self, new_data: list[tuple[int, int, int]]) -> None:
        """Inform the trigger about a batch of new data.

        PeriodicEvalTrigger will trigger on the first sample after the specified point in time.

        Side Effects:
            The trigger appends EvalRequests internally for  all points in time where evaluation is required.
        """

        if len(new_data) == 0 or self.unit_type == "epoch":
            return

        if self.unit_type == "samples":
            index_in_batch = 0
            while (len(new_data) - index_in_batch) >= self.remaining_samples_for_next_trigger:
                index_in_batch = index_in_batch + self.remaining_samples_for_next_trigger
                next_trigger_index = self.sample_counter + index_in_batch

                self.remaining_samples_for_next_trigger = self.every

                if (
                    len(self.evaluation_backlog) > 0
                    and self.evaluation_backlog[-1].sample_timestamp == new_data[index_in_batch][1]
                ):
                    continue

                self.evaluation_backlog.append(
                    EvalCandidate(
                        sample_index=next_trigger_index,
                        sample_id=new_data[index_in_batch][0],
                        sample_timestamp=new_data[index_in_batch][1],
                    )
                )

        elif self.unit_type == "epoch":
            pass  # already pre-determined with config

        else:
            raise ValueError(f"Invalid mode: {self.unit_type}")

        # Update trackers
        self.sample_counter += len(new_data)
