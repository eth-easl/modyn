from modyn.config.schema.pipeline import PeriodicEvalTriggerConfig


def test_periodic_eval_trigger() -> None:
    # test epoch based trigger
    trigger = PeriodicEvalTrigger(PeriodicEvalTriggerConfig(every="5s", start_timestamp=0, end_timestamp=20))

    assert trigger.evaluation_backlog == [
        # TODO: add intervals
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=0),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=5),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=10),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=15),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=20),
    ]
