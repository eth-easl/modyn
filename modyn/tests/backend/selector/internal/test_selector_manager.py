# pylint: disable=no-value-for-parameter,redefined-outer-name
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.selector.internal.selector_manager import SelectorManager
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.selector import Selector


class MockStrategy(AbstractSelectionStrategy):
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def _on_trigger(self) -> list[tuple[str, float]]:
        return []

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        pass

    def _reset_state(self) -> None:
        pass


def noop_init_metadata_db(self):
    pass


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test_init():
    selec = SelectorManager({})
    assert selec._next_pipeline_id == 0


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "_instantiate_strategy")
def test_register_pipeline(test__instantiate_strategy: MagicMock):
    selec = SelectorManager({})
    test__instantiate_strategy.return_value = MockStrategy()

    assert len(selec._selectors) == 0

    assert selec.register_pipeline(42, "{}") == 0
    assert len(selec._selectors) == 1

    assert selec._next_pipeline_id == 1

    assert isinstance(selec._selectors[0]._strategy, MockStrategy)

    with pytest.raises(ValueError):
        selec.register_pipeline(0, "strat")


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights(
    selector_get_sample_keys_and_weight: MagicMock, test__instantiate_strategy: MagicMock
):
    selec = SelectorManager({})
    test__instantiate_strategy.return_value = MockStrategy()
    pipe_id = selec.register_pipeline(2, "{}")

    with pytest.raises(ValueError):
        selec.get_sample_keys_and_weights(pipe_id + 1, 0, 0)

    with pytest.raises(ValueError):
        selec.get_sample_keys_and_weights(pipe_id, 0, 2)

    selector_get_sample_keys_and_weight.return_value = [("a", 1.0), ("b", 1.0)]

    assert selec.get_sample_keys_and_weights(0, 0, 0) == [("a", 1.0), ("b", 1.0)]

    selector_get_sample_keys_and_weight.assert_called_once_with(0, 0)


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "inform_data")
def test_inform_data(selector_inform_data: MagicMock, test__instantiate_strategy: MagicMock):
    selec = SelectorManager({})
    test__instantiate_strategy.return_value = MockStrategy()

    with pytest.raises(ValueError):
        selec.inform_data(0, ["a"], [0], [0])

    pipe_id = selec.register_pipeline(2, "{}")
    selector_inform_data.return_value = None

    selec.inform_data(pipe_id, ["a"], [0], [0])

    selector_inform_data.assert_called_once_with(["a"], [0], [0])


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "inform_data_and_trigger")
def test_inform_data_and_trigger(selector_inform_data_and_trigger: MagicMock, test__instantiate_strategy: MagicMock):
    selec = SelectorManager({})
    test__instantiate_strategy.return_value = MockStrategy()

    with pytest.raises(ValueError):
        selec.inform_data_and_trigger(0, ["a"], [0], [0])

    pipe_id = selec.register_pipeline(2, "{}")
    selector_inform_data_and_trigger.return_value = None

    selec.inform_data_and_trigger(pipe_id, ["a"], [0], [0])

    selector_inform_data_and_trigger.assert_called_once_with(["a"], [0], [0])


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test__instantiate_strategy():
    pass  # TODO(MaxiBoether): write this (test that limit was set + after newdatastrat is there)
