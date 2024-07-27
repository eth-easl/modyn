import pytest
from modyn.config.schema.pipeline import SlicingEvalStrategyConfig
from pydantic import TypeAdapter, ValidationError


def test_slicing_eval_strategy_config():
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
