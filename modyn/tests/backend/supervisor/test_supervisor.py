# pylint: disable=unused-argument
from modyn.backend.supervisor import Supervisor
from modyn.backend.supervisor.internal.grpc_handler import GRPCHandler
from unittest.mock import patch
import unittest.mock
import typing
import pytest
import time

def get_minimal_pipeline_config() -> dict:
    return {'pipeline': {'name': 'Test'},
            'model': {'id': 'ResNet18'},
            'training': {'gpus': 1},
            'data': {'dataset_id': 'test'},
            'trigger': {'id': 'DataAmountTrigger', 'trigger_config': {'every': 1}}}


def get_minimal_system_config() -> dict:
    return {}


def noop_constructor_mock(self, pipeline_config: dict, modyn_config: dict,  # pylint: disable=unused-argument
                          replay_at: typing.Optional[int]) -> None:  # pylint: disable=unused-argument
    pass


@patch.object(GRPCHandler, 'init_storage', return_value=None)
@patch.object(GRPCHandler, 'connection_established', return_value=True)
@patch.object(GRPCHandler, 'dataset_available', return_value=True)
def get_non_connecting_supervisor(test_dataset_available,  # pylint: disable=redefined-outer-name
                                  test_connection_established, test_init_storage) -> Supervisor:
    supervisor = Supervisor(get_minimal_pipeline_config(),
                            get_minimal_system_config(), None)

    return supervisor


def test_initialization() -> None:
    get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter


@patch.object(GRPCHandler, 'init_storage', return_value=None)
@patch.object(GRPCHandler, 'connection_established', return_value=False)
@patch.object(GRPCHandler, 'dataset_available', return_value=False)
def test_constructor_throws_on_invalid_system_config(test_dataset_available,  # pylint: disable=redefined-outer-name
                                                     test_connection_established, test_init_storage) -> None:
    with pytest.raises(ValueError, match="Invalid system configuration"):
        Supervisor(get_minimal_pipeline_config(), {}, None)


@patch.object(GRPCHandler, 'init_storage', return_value=None)
@patch.object(GRPCHandler, 'connection_established', return_value=True)
@patch.object(GRPCHandler, 'dataset_available', return_value=True)
def test_constructor_throws_on_invalid_pipeline_config(test_dataset_available,  # pylint: disable=redefined-outer-name
                                                       test_connection_established, test_init_storage) -> None:
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        Supervisor({}, get_minimal_system_config(), None)


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_validate_pipeline_config_schema():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_schema()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config_schema()

    # Check that an unknown model gets accepted because it has the correct schema
    # Semantic validation is done in another method
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config['model']['id'] = "UnknownModel"
    assert sup.validate_pipeline_config_schema()


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_validate_pipeline_config_content():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

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
    sup.pipeline_config['model']['id'] = "UnknownModel"
    assert not sup.validate_pipeline_config_content()

    # Check that training without GPUs gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config['training']['gpus'] = 0
    assert not sup.validate_pipeline_config_content()


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_validate_pipeline_config():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

    # Check that our minimal pipeline config gets accepted
    sup.pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config()

    # Check that an empty pipeline config gets rejected
    sup.pipeline_config = {}
    assert not sup.validate_pipeline_config()

    # Check that an unknown model gets rejected
    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config['model']['id'] = "UnknownModel"
    assert not sup.validate_pipeline_config()


@patch.object(GRPCHandler, 'dataset_available', lambda self, did: did == "existing")
def test_dataset_available():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.dataset_available()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.dataset_available()


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_trainer_available():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    assert sup.trainer_available()


@patch.object(GRPCHandler, 'dataset_available', lambda self, did: did == "existing")
def test_validate_system():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    # TODO(MaxiBoether): implement a better test when trainer_available is implemented
    assert sup.trainer_available()

    sup.pipeline_config = get_minimal_pipeline_config()
    sup.pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.validate_system()

    sup.pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.validate_system()


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_wait_for_new_data():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    
    sup.force_exit = True

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.wait_for_new_data(0)


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_initial_pass():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.initial_pass()


def test_replay_data():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.replay_at = 0
    sup.replay_data()


@patch.object(Supervisor, '__init__', noop_constructor_mock)
def test_end_pipeline():
    sup = Supervisor(get_minimal_pipeline_config(),
                     get_minimal_system_config(), None)

    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup.end_pipeline()


def test_pipeline():
    # TODO(MaxiBoether): implement a real test when func is implemented.
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.force_exit = True
    sup.pipeline()
