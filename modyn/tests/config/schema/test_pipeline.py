import pytest
from modyn.config.schema.pipeline import EvalStrategyModel, MatrixEvalStrategyConfig, OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval_strategies import OffsetEvalStrategy
from pydantic import TypeAdapter, ValidationError


def test_eval_strategy_model():
    with pytest.raises(ValidationError):
        TypeAdapter(EvalStrategyModel).validate_python({"name": "unknown"})


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
