import pytest
from pydantic import TypeAdapter, ValidationError

from modyn.config.schema.pipeline import EvalStrategyConfig, EvaluationConfig


def test_eval_strategy_model() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(EvalStrategyConfig).validate_python({"type": "unknown"})


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
