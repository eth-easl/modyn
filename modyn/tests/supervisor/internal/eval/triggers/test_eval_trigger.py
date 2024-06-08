import pandas as pd
from modyn.supervisor.internal.eval.triggers.eval_trigger import EvalCandidate, EvalRequest, EvalTrigger


def test_gen_pipeline_config():
    trigger_dataframe = pd.DataFrame(
        {
            "trigger_id": [1, 2, 3, 6, 7, 8],
            "training_id": [11, 12, 13, 16, 17, 18],
            "id_model": [21, 22, 23, 26, 27, 28],
            "first_timestamp": [0, 2, 3, 14, 20, 26],
            "last_timestamp": [0, 2, 3, 14, 20, 26],
        }
    )

    eval_trigger = EvalTrigger()
    eval_trigger.evaluation_backlog = [
        EvalCandidate(sample_timestamp=10),
        EvalCandidate(sample_timestamp=20),
        EvalCandidate(sample_timestamp=25),
        EvalCandidate(sample_timestamp=26),
        EvalCandidate(sample_timestamp=27),
    ]

    eval_requests = eval_trigger.get_eval_requests(trigger_dataframe, build_matrix=False)
    assert len(eval_requests) == len(eval_trigger.evaluation_backlog)

    # now assert the actual values in the full cross product (matrix mode)
    expected_eval_requests = [
        EvalRequest(
            trigger_id=1, training_id=11, model_id=21, most_recent_model=False, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=2, training_id=12, model_id=22, most_recent_model=False, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=3, training_id=13, model_id=23, most_recent_model=True, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=6, training_id=16, model_id=26, most_recent_model=False, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=7, training_id=17, model_id=27, most_recent_model=False, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=8, training_id=18, model_id=28, most_recent_model=False, interval_start=10, interval_end=10
        ),
        EvalRequest(
            trigger_id=1, training_id=11, model_id=21, most_recent_model=False, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=2, training_id=12, model_id=22, most_recent_model=False, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=3, training_id=13, model_id=23, most_recent_model=False, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=6, training_id=16, model_id=26, most_recent_model=False, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=7, training_id=17, model_id=27, most_recent_model=True, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=8, training_id=18, model_id=28, most_recent_model=False, interval_start=20, interval_end=20
        ),
        EvalRequest(
            trigger_id=1, training_id=11, model_id=21, most_recent_model=False, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=2, training_id=12, model_id=22, most_recent_model=False, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=3, training_id=13, model_id=23, most_recent_model=False, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=6, training_id=16, model_id=26, most_recent_model=False, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=7, training_id=17, model_id=27, most_recent_model=True, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=8, training_id=18, model_id=28, most_recent_model=False, interval_start=25, interval_end=25
        ),
        EvalRequest(
            trigger_id=1, training_id=11, model_id=21, most_recent_model=False, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=2, training_id=12, model_id=22, most_recent_model=False, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=3, training_id=13, model_id=23, most_recent_model=False, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=6, training_id=16, model_id=26, most_recent_model=False, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=7, training_id=17, model_id=27, most_recent_model=False, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=8, training_id=18, model_id=28, most_recent_model=True, interval_start=26, interval_end=26
        ),
        EvalRequest(
            trigger_id=1, training_id=11, model_id=21, most_recent_model=False, interval_start=27, interval_end=27
        ),
        EvalRequest(
            trigger_id=2, training_id=12, model_id=22, most_recent_model=False, interval_start=27, interval_end=27
        ),
        EvalRequest(
            trigger_id=3, training_id=13, model_id=23, most_recent_model=False, interval_start=27, interval_end=27
        ),
        EvalRequest(
            trigger_id=6, training_id=16, model_id=26, most_recent_model=False, interval_start=27, interval_end=27
        ),
        EvalRequest(
            trigger_id=7, training_id=17, model_id=27, most_recent_model=False, interval_start=27, interval_end=27
        ),
        EvalRequest(
            trigger_id=8, training_id=18, model_id=28, most_recent_model=True, interval_start=27, interval_end=27
        ),
    ]

    assert eval_trigger.get_eval_requests(trigger_dataframe, build_matrix=True) == expected_eval_requests
