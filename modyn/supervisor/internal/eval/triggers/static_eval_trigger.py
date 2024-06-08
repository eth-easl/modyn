from modyn.config.schema.pipeline.evaluation import StaticEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalTrigger


class StaticEvalTrigger(EvalTrigger):

    def __init__(self, config: StaticEvalTriggerConfig) -> None:
        super().__init__(
            backlog=[
                EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=epoch) for epoch in sorted(config.at)
            ]
        )
