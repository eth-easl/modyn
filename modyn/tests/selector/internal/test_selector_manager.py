# pylint: disable=no-value-for-parameter,redefined-outer-name
import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from modyn.metadata_database.models import Pipeline
from modyn.selector.internal.selector_manager import SelectorManager
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.selector.selector import Selector


def get_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "user",
            "password": "pw",
            "database": "db",
            "host": "derhorst",
            "port": "1337",
        },
        "selector": {"keys_in_selector_cache": 1000, "trigger_sample_directory": "/does/not/exist"},
    }


class MockStrategy(AbstractSelectionStrategy):
    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def _on_trigger(self) -> list[tuple[str, float]]:  # pylint: disable=unused-argument
        return []

    def inform_data(
        self, keys: list[str], timestamps: list[int], labels: list[int]
    ) -> None:  # pylint: disable=unused-argument
        pass

    def _reset_state(self) -> None:  # pylint: disable=unused-argument
        pass


class MockDatabaseConnection:
    def __init__(self, modyn_config: dict):  # pylint: disable=super-init-not-called,unused-argument
        self.current_pipeline_id = 0
        self.session = MockSession()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception):
        pass


class MockSession:
    def get(self, some_type, pipeline_id) -> Optional[Pipeline]:  # pylint: disable=unused-argument
        if pipeline_id == 0:
            return Pipeline(num_workers=2, selection_strategy="{}")
        return None


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        SelectorManager(config)


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
def test_init_throws_non_existing_dir():
    with pytest.raises(ValueError):
        SelectorManager(get_modyn_config())


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights(
    selector_get_sample_keys_and_weight: MagicMock, test__instantiate_strategy: MagicMock
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        selec = SelectorManager(config)

        test__instantiate_strategy.return_value = MockStrategy()
        pipe_id = 0

        with pytest.raises(ValueError):
            # Non existing pipeline
            selec.get_sample_keys_and_weights(pipe_id + 1, 0, 0, 0)

        with pytest.raises(ValueError):
            # Too many workers
            selec.get_sample_keys_and_weights(pipe_id, 0, 2, 0)

        selector_get_sample_keys_and_weight.return_value = [(10, 1.0), (11, 1.0)]

        assert selec.get_sample_keys_and_weights(0, 0, 0, 0) == [(10, 1.0), (11, 1.0)]

        selector_get_sample_keys_and_weight.assert_called_once_with(0, 0, 0)


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "inform_data")
def test_inform_data(selector_inform_data: MagicMock, test__instantiate_strategy: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        selec = SelectorManager(config)
        test__instantiate_strategy.return_value = MockStrategy()

        pipe_id = 0
        selector_inform_data.return_value = None

        selec.inform_data(pipe_id, [10], [0], [0])

        with pytest.raises(ValueError):
            selec.inform_data(pipe_id + 1, [10], [0], [0])

        selector_inform_data.assert_called_once_with([10], [0], [0])


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "inform_data_and_trigger")
def test_inform_data_and_trigger(selector_inform_data_and_trigger: MagicMock, test__instantiate_strategy: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        selec = SelectorManager(config)
        test__instantiate_strategy.return_value = MockStrategy()

        pipe_id = 0
        selector_inform_data_and_trigger.return_value = None

        selec.inform_data_and_trigger(pipe_id, [10], [0], [0])

        with pytest.raises(ValueError):
            selec.inform_data_and_trigger(pipe_id + 1, [10], [0], [0])

        selector_inform_data_and_trigger.assert_called_once_with([10], [0], [0])


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "get_available_labels")
def test_get_available_labels(selector_get_available_labels: MagicMock, test__instantiate_strategy: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        selector = SelectorManager(config)
        test__instantiate_strategy.return_value = MockStrategy()

        pipe_id = 0
        selector_get_available_labels.return_value = None

        selector.get_available_labels(pipe_id)
        selector_get_available_labels.assert_called_once_with()

        with pytest.raises(ValueError):
            selector.get_available_labels(pipe_id + 1)


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
def test_init_selector_manager_with_existing_trigger_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "test"), "w", encoding="utf-8") as file:
            file.write("test")

        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir

        with pytest.raises(ValueError):
            SelectorManager(config)


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
def test__instantiate_strategy():
    pass  # TODO(MaxiBoether): Implement this at a later point


@patch("modyn.selector.internal.selector_manager.MetadataDatabaseConnection", MockDatabaseConnection)
@patch.object(SelectorManager, "_instantiate_strategy")
@patch.object(Selector, "get_number_of_samples")
def test_get_number_of_samples(selector_get_number_of_samples: MagicMock, test__instantiate_strategy: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        selec = SelectorManager(config)
        test__instantiate_strategy.return_value = MockStrategy()

        pipe_id = 0
        selector_get_number_of_samples.return_value = 12

        assert selec.get_number_of_samples(pipe_id, 21) == 12

        with pytest.raises(ValueError):
            selec.get_number_of_samples(pipe_id + 1, 0)

        selector_get_number_of_samples.assert_called_once_with(21)
