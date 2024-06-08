import pandas as pd
from modyn.config.schema.pipeline.evaluation import StaticEvalTriggerConfig
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalRequest
from modyn.supervisor.internal.eval.triggers.static_eval_trigger import StaticEvalTrigger


def test_static_eval_trigger() -> None:
    # test epoch based trigger
    config = StaticEvalTriggerConfig(at=[3, 6, 7, 8, 11], start_timestamp=0)
    trigger = StaticEvalTrigger(config)

    assert trigger.evaluation_backlog == [
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=3),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=6),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=7),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=8),
        EvalCandidate(sample_index=None, sample_id=None, sample_timestamp=11),
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

    assert len(eval_requests) == 5

    # timestamps from the evaluation_backlog, and model_ids from trigger_dataframe
    assert eval_requests[0] == EvalRequest(
        trigger_id=2, training_id=12, model_id=22, most_recent_model=True, interval_start=3, interval_end=3
    )
    assert eval_requests[1] == EvalRequest(
        trigger_id=6, training_id=16, model_id=26, most_recent_model=True, interval_start=6, interval_end=6
    )
    assert eval_requests[2] == EvalRequest(
        trigger_id=7, training_id=17, model_id=27, most_recent_model=True, interval_start=7, interval_end=7
    )
    assert eval_requests[3] == EvalRequest(
        trigger_id=7, training_id=17, model_id=27, most_recent_model=True, interval_start=8, interval_end=8
    )
    assert eval_requests[4] == EvalRequest(
        trigger_id=8, training_id=18, model_id=28, most_recent_model=True, interval_start=11, interval_end=11
    )
