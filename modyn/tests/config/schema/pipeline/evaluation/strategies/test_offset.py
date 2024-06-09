import pytest
from pydantic import TypeAdapter, ValidationError

from modyn.config.schema.pipeline import OffsetEvalStrategyConfig
from modyn.supervisor.internal.eval.strategies import OffsetEvalStrategy


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

