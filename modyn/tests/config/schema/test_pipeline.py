import pytest
from modyn.config.schema.pipeline import EvaluationConfig, OffsetEvalStrategyConfig, SlicingEvalStrategyConfig
from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from pydantic import TypeAdapter, ValidationError


def test_matrix_eval_strategy_config():
    with pytest.raises(ValidationError):
        TypeAdapter(SlicingEvalStrategyConfig).validate_python(
            {
                "type": "SlicingEvalStrategy",
                "eval_every": "100s",
                "eval_start_from": -100,
                "eval_end_at": 300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(SlicingEvalStrategyConfig).validate_python(
            {
                "type": "SlicingEvalStrategy",
                "eval_every": "100s",
                "eval_start_from": 100,
                "eval_end_at": -300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(SlicingEvalStrategyConfig).validate_python(
            {
                "type": "SlicingEvalStrategy",
                "eval_every": "100s",
                "eval_start_from": 300,
                "eval_end_at": 100,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(SlicingEvalStrategyConfig).validate_python(
            {
                "type": "SlicingEvalStrategy",
                "eval_every": "100s",
                "eval_start_from": 300,
                "eval_end_at": 300,
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(SlicingEvalStrategyConfig).validate_python(
            {
                "type": "SlicingEvalStrategy",
                "eval_every": "100s10d",
                "eval_start_from": 100,
                "eval_end_at": 300,
            }
        )


def test_offset_eval_strategy_config():
    with pytest.raises(ValidationError):
        TypeAdapter(OffsetEvalStrategyConfig).validate_python(
            {
                "offsets": [],
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(OffsetEvalStrategyConfig).validate_python(
            {
                "offsets": [0],
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(OffsetEvalStrategyConfig).validate_python(
            {
                "offsets": ["+inf"],
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(OffsetEvalStrategyConfig).validate_python(
            {
                "offsets": ["10d10s"],
            }
        )

    TypeAdapter(OffsetEvalStrategyConfig).validate_python(
        {
            "offsets": [OffsetEvalStrategy.INFINITY, OffsetEvalStrategy.NEGATIVE_INFINITY, "10s", "10d", "-10s", "0s"],
        }
    )


def test_evaluation_config_duplicate_dataset_ids() -> None:
    minimal_evaluation_config = {
        "handlers": [
            {
                "execution_time": "after_training",
                "strategy": {
                    "type": "SlicingEvalStrategy",
                    "eval_every": "100s",
                    "eval_start_from": 0,
                    "eval_end_at": 300,
                },
                "models": "matrix",
                "datasets": ["MNIST_eval"],
            }
        ],
        "device": "cpu",
        "datasets": [
            {
                "dataset_id": "MNIST_eval",
                "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
                "dataloader_workers": 2,
                "batch_size": 64,
                "metrics": [{"name": "Accuracy"}],
            }
        ],
    }
    # verify that the correct config can pass
    TypeAdapter(EvaluationConfig).validate_python(minimal_evaluation_config)

    eval_dataset_config = minimal_evaluation_config["datasets"][0]
    # duplicate dataset_id
    minimal_evaluation_config["datasets"].append(eval_dataset_config)
    with pytest.raises(ValidationError):
        TypeAdapter(EvaluationConfig).validate_python(minimal_evaluation_config)
