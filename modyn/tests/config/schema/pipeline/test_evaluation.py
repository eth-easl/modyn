import pytest
from modyn.config.schema.pipeline import MatrixEvalStrategyConfig
from modyn.config.schema.pipeline.evaluation import EvalStrategyConfig, EvaluationConfig, IntervalEvalStrategyConfig
from pydantic import TypeAdapter, ValidationError


def test_eval_strategy_model():
    with pytest.raises(ValidationError):
        TypeAdapter(EvalStrategyConfig).validate_python({"name": "unknown"})


def test_matrix_eval_strategy_config():
    with pytest.raises(ValidationError):
        TypeAdapter(MatrixEvalStrategyConfig).validate_python(
            {
                "eval_every": "100s",
                "eval_start_from": -100,
                "eval_end_at": 300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(MatrixEvalStrategyConfig).validate_python(
            {
                "eval_every": "100s",
                "eval_start_from": 100,
                "eval_end_at": -300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(MatrixEvalStrategyConfig).validate_python(
            {
                "eval_every": "100s",
                "eval_start_from": 300,
                "eval_end_at": 100,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(MatrixEvalStrategyConfig).validate_python(
            {
                "eval_every": "100s",
                "eval_start_from": 300,
                "eval_end_at": 300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(MatrixEvalStrategyConfig).validate_python(
            {
                "eval_every": "100s10d",
                "eval_start_from": 100,
                "eval_end_at": 300,
            }
        )


VALID_INTERVALS = [
    ("[-inf;+inf]", "-inf", "d", True, "+inf", "d", True),
    ("[-inf,+inf]", "-inf", "d", True, "+inf", "d", True),
    ("[-inf,+0d]", "-inf", "d", True, "+0", "d", True),
    ("[-inf,-0d]", "-inf", "d", True, "-0", "d", True),
    ("[-inf,0d]", "-inf", "d", True, "0", "d", True),
    ("[+0d,+inf]", "+0", "d", True, "+inf", "d", True),
    ("[-0d,+inf]", "-0", "d", True, "+inf", "d", True),
    ("[+0d,+inf]", "+0", "d", True, "+inf", "d", True),
    ("[-20d,-15h]", "-20", "d", True, "-15", "h", True),
    ("[+10s,+20m]", "+10", "s", True, "+20", "m", True),
    ("( +10s, +20m]", "+10", "s", False, "+20", "m", True),
    ("[ +10s, +20m)", "+10", "s", True, "+20", "m", False),
    ("( +10s, +20m)", "+10", "s", False, "+20", "m", False),
]
"""interval string, expected: left, left_unit, left_bound_inclusive, right, right_unit, right_bound_inclusive"""


@pytest.mark.parametrize(
    "interval,exp_left,exp_left_unit,exp_left_bound_inclusive,exp_right,exp_right_unit,exp_right_bound_inclusive",
    VALID_INTERVALS,
)
def test_interval_eval_strategy_config_valid_intervals(
    interval: str,
    exp_left: str,
    exp_left_unit: str,
    exp_left_bound_inclusive: str,
    exp_right: str,
    exp_right_unit: str,
    exp_right_bound_inclusive: str,
) -> None:
    res = IntervalEvalStrategyConfig.model_validate({"interval": interval})
    assert res.left == exp_left
    assert res.left_unit == exp_left_unit
    assert res.left_bound_inclusive == exp_left_bound_inclusive
    assert res.right == exp_right
    assert res.right_unit == exp_right_unit
    assert res.right_bound_inclusive == exp_right_bound_inclusive


INVALID_INTERVALS = [
    ("[+inf,-inf"),
    ("{-1,1}"),
    ("{-11}"),
    ("{-11,}"),
    ("{+3d,+1d}"),
    ("{+3d,+1d}"),
    ("{+inf,+3d}"),
    ("{+3d,-inf}"),
]


@pytest.mark.parametrize("interval", INVALID_INTERVALS)
def test_interval_eval_strategy_config_invalid_intervals(interval: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(IntervalEvalStrategyConfig).validate_python({"interval": interval})

    minimal_evaluation_config = {
        "handlers": [
            {
                "strategy": {
                    "type": "MatrixEvalStrategy",
                    "eval_every": "100s",
                    "eval_start_from": 0,
                    "eval_end_at": 300,
                },
                "trigger": {"mode": "after_training"},
                "datasets": ["mnist_eval"],
            }
        ],
        "device": "cpu",
        "datasets": {
            "mnist_eval": {
                "dataset_id": "MNIST_eval",
                "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                "dataloader_workers": 2,
                "batch_size": 64,
                "metrics": [{"name": "Accuracy"}],
            }
        },
    }
    # verify that the correct config can pass
    TypeAdapter(EvaluationConfig).validate_python(minimal_evaluation_config)

    eval_dataset_config = minimal_evaluation_config["datasets"]["mnist_eval"]  # type: ignore
    # duplicate dataset_id
    minimal_evaluation_config["datasets"]["mnist_eval_2"] = eval_dataset_config  # type: ignore
    with pytest.raises(ValidationError):
        TypeAdapter(EvaluationConfig).validate_python(minimal_evaluation_config)
