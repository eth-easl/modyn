import pytest
from modyn.supervisor.internal.eval_strategies import PeriodicalEvalStrategy


def get_minimal_eval_strategies_config() -> dict:
    return {
        "eval_every": "100s",
        "eval_start_from": 0,
        "eval_end_at": 300,
    }


def test_initialization() -> None:
    eval_strategy = PeriodicalEvalStrategy(get_minimal_eval_strategies_config())
    assert eval_strategy.eval_every == 100
    assert eval_strategy.eval_start_from == 0
    assert eval_strategy.eval_end_at == 300


def test_init_fails_if_invalid() -> None:
    config = get_minimal_eval_strategies_config()
    config["eval_every"] = "0s"
    with pytest.raises(AssertionError, match="eval_every must be greater than 0"):
        PeriodicalEvalStrategy(config)
    config["eval_every"] = "10s"
    config["eval_start_from"] = 400
    with pytest.raises(AssertionError, match="eval_start_from must be less than eval_end_at"):
        PeriodicalEvalStrategy(config)


def test_get_eval_interval() -> None:
    config = get_minimal_eval_strategies_config()
    eval_strategy = PeriodicalEvalStrategy(config)
    assert list(eval_strategy.get_eval_interval(0, 0)) == [
        (0, 100),
        (100, 200),
        (200, 300),
    ]

    config["eval_start_from"] = 50
    eval_strategy = PeriodicalEvalStrategy(config)
    assert list(eval_strategy.get_eval_interval(0, 0)) == [
        (50, 150),
        (150, 250),
        (250, 300),
    ]
