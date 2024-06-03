import pandas as pd
from modyn.config.schema.pipeline.evaluation import StaticEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalRequest
from modyn.supervisor.internal.eval.triggers.static_eval_trigger import StaticEvalTrigger


def test_static_eval_trigger() -> None:
    # test epoch based trigger
    config = StaticEvalTriggerConfig(unit="epoch", at=[3, 6, 7, 8, 11], start_timestamp=0)
    trigger = StaticEvalTrigger(config)

    new_data = [
        (11, 1, 0),
        (12, 2, -1),
        (13, 3, -1),
        (14, 4, -1),
        (15, 5, -1),
        (16, 9, -1),
        (17, 9, -1),
        (18, 11, -1),
    ]
    trigger.inform(new_data=new_data)

    assert trigger.evaluation_backlog == [
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=3),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=6),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=7),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=8),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=11),
    ]

    # test sample_index based trigger
    config = StaticEvalTriggerConfig(unit="sample_index", at=[2, 3, 5, 6], start_timestamp=0)
    trigger = StaticEvalTrigger(config)
    trigger.inform(new_data=new_data)

    assert trigger.evaluation_backlog == [
        EvalCandidate(sample_index=2, sample_id=13, sample_timestamp=3),
        EvalCandidate(sample_index=3, sample_id=14, sample_timestamp=4),
        EvalCandidate(sample_index=5, sample_id=16, sample_timestamp=9),  # handles triggers at indexes 6, 7, 8
        # skipped as would use the same timestamp than the previous one
    ]

    # test get_eval_requests
    trigger_dataframe = pd.DataFrame(
        {
            "trigger_id": [1, 2, 3, 6, 7, 8],
            "training_id": [11, 12, 13, 16, 17, 18],
            "id_model": [21, 22, 23, 26, 27, 28],
            "first_timestamp": [0, 1, 3, 6, 7, 8],
            "last_timestamp": [0, 2, 4, 6, 7, 9],
        }
    )
    eval_requests = trigger.get_eval_requests(trigger_dataframe, build_matrix=False)

    assert len(eval_requests) == 3

    # timestamps from the evaluation_backlog, and model_ids from trigger_dataframe
    assert eval_requests[0] == EvalRequest(
        trigger_id=2, training_id=12, model_id=22, most_recent_model=True, interval_start=3, interval_end=3
    )
    assert eval_requests[1] == EvalRequest(
        trigger_id=3, training_id=13, model_id=23, most_recent_model=True, interval_start=4, interval_end=4
    )
    assert eval_requests[2] == EvalRequest(
        trigger_id=8, training_id=18, model_id=28, most_recent_model=True, interval_start=9, interval_end=9
    )
