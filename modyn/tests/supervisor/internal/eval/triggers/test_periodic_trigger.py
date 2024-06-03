from modyn.config.schema.pipeline.evaluation import PeriodicEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate
from modyn.supervisor.internal.eval.triggers.periodic_eval_trigger import PeriodicEvalTrigger


def test_periodic_eval_trigger() -> None:
    # test epoch based trigger
    trigger = PeriodicEvalTrigger(PeriodicEvalTriggerConfig(every="5s", start_timestamp=0, end_timestamp=20))

    new_data = [
        (11, 1, 0),
        (12, 2, -1),
        (13, 3, -1),
        (14, 4, -1),
        (15, 5, -1),
        (16, 6, -1),
        (17, 20, -1),
        (18, 21, -1),
        (19, 21, -1),
        (20, 21, -1),
        (21, 21, -1),
        (22, 21, -1),
        (23, 21, -1),
        (24, 21, -1),
        (25, 40, -1),
        (26, 41, -1),
        (27, 41, -1),
        (28, 41, -1),
    ]
    trigger.inform(new_data=new_data)

    assert trigger.evaluation_backlog == [
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=0),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=5),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=10),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=15),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=20),
    ]

    # test sample_index based trigger
    trigger = PeriodicEvalTrigger(PeriodicEvalTriggerConfig(every="4"))
    trigger.inform(new_data=new_data)

    assert trigger.evaluation_backlog == [
        EvalCandidate(sample_index=0, sample_id=11, sample_timestamp=1),
        EvalCandidate(sample_index=4, sample_id=15, sample_timestamp=5),
        EvalCandidate(sample_index=8, sample_id=19, sample_timestamp=21),
        # sample 12 skipped (timestamp already seen)
        EvalCandidate(sample_index=16, sample_id=27, sample_timestamp=41),
    ]
