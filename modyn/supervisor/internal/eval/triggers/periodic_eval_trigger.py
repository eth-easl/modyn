from modyn.config.schema.pipeline.evaluation import PeriodicEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalTrigger


class PeriodicEvalTrigger(EvalTrigger):

    def __init__(self, config: PeriodicEvalTriggerConfig) -> None:
        super().__init__(
            backlog=[
                EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=epoch)
                for epoch in range(config.start_timestamp, config.end_timestamp + 1, config.every_sec)
            ]
        )
