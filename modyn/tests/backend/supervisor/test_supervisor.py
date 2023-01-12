# pylint: disable=unused-argument,redefined-outer-name
import typing
from unittest.mock import MagicMock, patch

import pytest
from modyn.backend.supervisor import Supervisor
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from modyn.backend.supervisor.internal.triggers.amounttrigger import DataAmountTrigger


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "training": {
            "gpus": 1,
            "dataloader_workers": 1,
            "learning_rate": 0.1,
            "batch_size": 42,
            "strategy": "finetune",
            "initial_model": "random",
            "initial_pass": {"activated": False},
        },
        "data": {"dataset_id": "test"},
        "trigger": {
            "id": "DataAmountTrigger",
            "trigger_config": {"data_points_for_trigger": 1},
        },
    }


def get_minimal_system_config() -> dict:
    return {}


def noop_constructor_mock(self, pipeline_config: dict, modyn_config: dict, replay_at: typing.Optional[int]) -> None:
    pass


def sleep_mock(duration: int):
    raise KeyboardInterrupt


@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=True)
def get_non_connecting_supervisor(test_dataset_available, test_connection_established, test_init_storage) -> Supervisor:
    supervisor = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    return supervisor


def test_initialization() -> None:
    get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter


@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=False)
def test_constructor_throws_on_invalid_system_config(
    test_dataset_available, test_connection_established, test_init_storage
) -> None:
    with pytest.raises(ValueError, match="Invalid system configuration"):
        Supervisor(get_minimal_pipeline_config(), {}, None)


@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=True)
def test_constructor_throws_on_invalid_pipeline_config(
    test_dataset_available, test_connection_established, test_init_storage
) -> None:
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        Supervisor({}, get_minimal_system_config(), None)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_schema():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_schema()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config_schema()

    # Check that an unknown model gets accepted because it has the correct schema
    # Semantic validation is done in another method
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert sup.validate_pipeline_config_schema()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__validate_training_options():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup._validate_training_options()

    # Check that training without GPUs gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["gpus"] = 0
    assert not sup._validate_training_options()

    # Check that training with an invalid batch size gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["batch_size"] = -1
    assert not sup._validate_training_options()

    # Check that training with an invalid strategy gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["strategy"] = "UnknownStrategy"
    assert not sup._validate_training_options()

    # Check that training with an invalid initial model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_model"] = "UnknownInitialModel"
    assert not sup._validate_training_options()

    # Check that training with an invalid reference for initial pass gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "UnknownRef"
    assert not sup._validate_training_options()

    # Check that training with an invalid amount for initial pass gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "amount"
    sup.pipeline_config["training"]["initial_pass"]["amount"] = 2
    assert not sup._validate_training_options()

    # Check that training with an valid amount for initial pass gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["initial_pass"]["activated"] = True
    sup.pipeline_config["training"]["initial_pass"]["reference"] = "amount"
    sup.pipeline_config["training"]["initial_pass"]["amount"] = 0.5
    assert sup._validate_training_options()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_content():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_content()

    # Check that an empty pipeline config throws an exception
    # because there is no model defined
    with pytest.raises(KeyError):
        sup.pipeline_config = {}
        assert not sup.validate_pipeline_config_content()

    # Check that an unknown model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config_content()

    # Check that an unknown trigger gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["trigger"]["id"] = "UnknownTrigger"
    assert not sup.validate_pipeline_config_content()

    # Check that training without GPUs gets rejected
    # (testing that _validate_training_options gets called)
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["training"]["gpus"] = 0
    assert not sup.validate_pipeline_config_content()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config()

    # Check that an unknown model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config()


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
def test_dataset_available():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.dataset_available()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.dataset_available()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_trainer_available():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    assert sup.trainer_available()


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
def test_validate_system():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    # TODO(MaxiBoether): implement a better test when trainer_available is implemented
    assert sup.trainer_available()

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.validate_system()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.validate_system()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__query_new_data_from_storage():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    assert not sup._query_new_data_from_storage(0)


@patch.object(Supervisor, "_query_new_data_from_storage", return_value=[("a", 42), ("b", 43)])
@patch.object(DataAmountTrigger, "inform", return_value=False, side_effect=KeyboardInterrupt)
def test_wait_for_new_data(test_inform: MagicMock, test__query_new_data_from_storage: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.wait_for_new_data(42)

    test__query_new_data_from_storage.assert_called_once_with(42)
    test_inform.assert_called_once_with([("a", 42), ("b", 43)])


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__on_trigger():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup._on_trigger()


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_initial_pass():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.initial_pass()


@patch.object(Supervisor, "_query_new_data_from_storage", return_value=[("a", 42), ("b", 43)])
@patch.object(DataAmountTrigger, "inform")
def test_replay_data(test_inform: MagicMock, test__query_new_data_from_storage: MagicMock):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.replay_at = 42
    sup.replay_data()

    test__query_new_data_from_storage.assert_called_once()
    test_inform.assert_called_once_with([("a", 42), ("b", 43)])


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_end_pipeline():
    sup = Supervisor(get_minimal_pipeline_config(), get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.end_pipeline()


@patch.object(Supervisor, "initial_pass")
@patch.object(Supervisor, "replay_data")
@patch.object(Supervisor, "wait_for_new_data")
@patch.object(Supervisor, "end_pipeline")
def test_non_experiment_pipeline(
    test_end_pipeline: MagicMock,
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_initial_pass: MagicMock,
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.experiment_mode = False
    sup.pipeline()

    test_initial_pass.assert_called_once()
    test_wait_for_new_data.assert_called_once()
    test_replay_data.assert_not_called()
    test_end_pipeline.assert_called_once()


@patch.object(Supervisor, "initial_pass")
@patch.object(Supervisor, "replay_data")
@patch.object(Supervisor, "wait_for_new_data")
@patch.object(Supervisor, "end_pipeline")
def test_experiment_pipeline(
    test_end_pipeline: MagicMock,
    test_wait_for_new_data: MagicMock,
    test_replay_data: MagicMock,
    test_initial_pass: MagicMock,
):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.experiment_mode = True
    sup.pipeline()

    test_initial_pass.assert_called_once()
    test_wait_for_new_data.assert_not_called()
    test_replay_data.assert_called_once()
    test_end_pipeline.assert_called_once()
