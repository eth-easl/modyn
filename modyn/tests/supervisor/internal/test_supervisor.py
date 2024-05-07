# pylint: disable=unused-argument,redefined-outer-name
import multiprocessing as mp
import os
import pathlib
import shutil
import time
from typing import Optional
from unittest import mock
from unittest.mock import patch

import pytest
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.grpc.enums import PipelineStatus
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor
from modyn.supervisor.internal.utils import PipelineInfo

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
SUPPORTED_EVAL_RESULT_WRITERS: dict = {"json": JsonResultWriter, "tensorboard": TensorboardResultWriter}
START_TIMESTAMP = 21
PIPELINE_ID = 42


def get_minimal_training_config() -> dict:
    return {
        "gpus": 1,
        "device": "cpu",
        "dataloader_workers": 1,
        "use_previous_model": True,
        "initial_model": "random",
        "learning_rate": 0.1,
        "batch_size": 42,
        "optimizers": [
            {"name": "default1", "algorithm": "SGD", "source": "PyTorch", "param_groups": [{"module": "model"}]},
        ],
        "optimization_criterion": {"name": "CrossEntropyLoss"},
        "checkpointing": {"activated": False},
        "selection_strategy": {"name": "NewDataStrategy", "maximum_keys_in_memory": 10},
    }


def get_minimal_evaluation_config() -> dict:
    return {
        "device": "cpu",
        "dataset": {
            "dataset_id": "MNIST_eval",
            "bytes_parser_function": "def bytes_parser_function(data: bytes) -> bytes:\n\treturn data",
            "dataloader_workers": 2,
            "batch_size": 64,
            "metrics": [{"name": "Accuracy"}],
        },
    }


def get_minimal_pipeline_config() -> dict:
    return {
        "pipeline": {"name": "Test"},
        "model": {"id": "ResNet18"},
        "model_storage": {"full_model_strategy": {"name": "PyTorchFullModel"}},
        "training": get_minimal_training_config(),
        "data": {"dataset_id": "test", "bytes_parser_function": "def bytes_parser_function(x):\n\treturn x"},
        "trigger": {"id": "DataAmountTrigger", "trigger_config": {"data_points_for_trigger": 1}},
    }


def get_minimal_system_config() -> dict:
    return {
        "supervisor": {
            "eval_directory": EVALUATION_DIRECTORY,
        }
    }


def noop_constructor_mock(self, modyn_config: dict) -> None:
    pass


def noop_init_metadata_db(self) -> None:
    pass


def noop_pipeline_executor_constructor_mock(
    self,
    start_timestamp: int,
    pipeline_id: int,
    modyn_config: dict,
    pipeline_config: dict,
    eval_directory: str,
    supervisor_supported_eval_result_writers: dict,
    pipeline_status_queue: mp.Queue,
    training_status_queue: mp.Queue,
    eval_status_queue: mp.Queue,
    start_replay_at: Optional[int] = None,
    stop_replay_at: Optional[int] = None,
    maximum_triggers: Optional[int] = None,
    evaluation_matrix: bool = False,
) -> None:
    pass


def noop() -> None:
    pass


def sleep_mock(duration: int):
    raise KeyboardInterrupt


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    if EVALUATION_DIRECTORY.is_dir():
        shutil.rmtree(EVALUATION_DIRECTORY)
    EVALUATION_DIRECTORY.mkdir(0o777)

    yield
    shutil.rmtree(EVALUATION_DIRECTORY)


class MockDatabaseConnection:
    def __init__(self, modyn_config: dict):  # pylint: disable=super-init-not-called,unused-argument
        self.current_pipeline_id = 0
        self.session = MockSession()

    # pylint: disable=unused-argument
    def register_pipeline(
        self,
        num_workers: int,
        model_class_name: str,
        model_config: str,
        amp: bool,
        selection_strategy: str,
        full_model_strategy: ModelStorageStrategyConfig,
        incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None,
        full_model_interval: Optional[int] = None,
    ) -> Optional[int]:
        pid = self.current_pipeline_id
        self.current_pipeline_id += 1
        return pid

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception):
        pass


class MockSession:
    def get(self, some_type, pipeline_id):  # pylint: disable=unused-argument
        return None


@patch.object(GRPCHandler, "init_selector", return_value=None)
@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch.object(GRPCHandler, "init_trainer_server", return_value=None)
@patch.object(GRPCHandler, "init_evaluator", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def get_non_connecting_supervisor(
    test_connection_established,
    test_init_evaluator,
    test_init_trainer_server,
    test_init_storage,
    test_init_selector,
) -> Supervisor:
    supervisor = Supervisor(get_minimal_system_config())
    supervisor.init_cluster_connection()

    return supervisor


def test_initialization() -> None:
    get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_schema():
    sup = Supervisor(get_minimal_system_config())

    # Check that our minimal pipeline config gets accepted
    pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_schema(pipeline_config)

    # Check that an empty pipeline config gets rejected
    pipeline_config = {}
    assert not sup.validate_pipeline_config_schema(pipeline_config)

    # Check that an unknown model gets accepted because it has the correct schema
    # Semantic validation is done in another method
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["model"]["id"] = "UnknownModel"
    assert sup.validate_pipeline_config_schema(pipeline_config)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__validate_training_options():
    sup = Supervisor(get_minimal_system_config())

    # Check that our minimal training config gets accepted
    training_config = get_minimal_training_config()
    assert sup._validate_training_options(training_config)

    # Check that training without GPUs gets rejected
    training_config = get_minimal_training_config()
    training_config["gpus"] = 0
    assert not sup._validate_training_options(training_config)

    # Check that training with an invalid batch size gets rejected
    training_config = get_minimal_training_config()
    training_config["batch_size"] = -1
    assert not sup._validate_training_options(training_config)

    # Check that training with an invalid dataloader amount gets rejected
    training_config = get_minimal_training_config()
    training_config["dataloader_workers"] = -1
    assert not sup._validate_training_options(training_config)

    # Check that training with an invalid strategy gets rejected
    training_config = get_minimal_training_config()
    training_config["selection_strategy"]["name"] = "UnknownStrategy"
    assert not sup._validate_training_options(training_config)

    # Check that training with an invalid initial model gets rejected
    training_config = get_minimal_training_config()
    training_config["initial_model"] = "UnknownInitialModel"
    assert not sup._validate_training_options(training_config)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test__validate_evaluation_options():
    sup = Supervisor(get_minimal_system_config())

    # Check that evaluation with an invalid batch size gets rejected
    evaluation_config = get_minimal_evaluation_config()
    evaluation_config["dataset"]["batch_size"] = -1
    assert not sup._validate_evaluation_options(evaluation_config)

    # Check that evaluation with an invalid dataloader amount gets rejected
    evaluation_config = get_minimal_evaluation_config()
    evaluation_config["dataset"]["dataloader_workers"] = -1
    assert not sup._validate_evaluation_options(evaluation_config)

    # Check that evaluation with invalid evaluation writer gets rejected
    evaluation_config = get_minimal_evaluation_config()
    evaluation_config["result_writers"] = ["json", "unknown", "unknown2"]
    assert not sup._validate_evaluation_options(evaluation_config)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config_content():
    sup = Supervisor(get_minimal_system_config())
    evaluation_matrix = False

    # Check that our minimal pipeline config gets accepted
    pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config_content(pipeline_config, evaluation_matrix)

    # Check that an empty pipeline config throws an exception
    # because there is no model defined
    with pytest.raises(KeyError):
        pipeline_config = {}
        assert not sup.validate_pipeline_config_content(pipeline_config, evaluation_matrix)

    # Check that an unknown model gets rejected
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config_content(pipeline_config, evaluation_matrix)

    # Check that an unknown trigger gets rejected
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["trigger"]["id"] = "UnknownTrigger"
    assert not sup.validate_pipeline_config_content(pipeline_config, evaluation_matrix)

    # Check that training without GPUs gets rejected
    # (testing that _validate_training_options gets called)
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["training"]["gpus"] = 0
    assert not sup.validate_pipeline_config_content(pipeline_config, evaluation_matrix)


@patch.object(Supervisor, "__init__", noop_constructor_mock)
def test_validate_pipeline_config():
    sup = Supervisor(get_minimal_system_config())
    evaluation_matrix = False

    # Check that our minimal pipeline config gets accepted
    pipeline_config = get_minimal_pipeline_config()
    assert sup.validate_pipeline_config(pipeline_config, evaluation_matrix)

    # Check that an empty pipeline config gets rejected
    pipeline_config = {}
    assert not sup.validate_pipeline_config(pipeline_config, evaluation_matrix)

    # Check that an unknown model gets rejected
    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["model"]["id"] = "UnknownModel"
    assert not sup.validate_pipeline_config(pipeline_config, evaluation_matrix)


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
def test_dataset_available():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.dataset_available(pipeline_config)

    pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.dataset_available(pipeline_config)


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def test_validate_system(test_trainer_server_available):
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    pipeline_config = get_minimal_pipeline_config()
    pipeline_config["data"]["dataset_id"] = "existing"
    assert sup.validate_system(pipeline_config)

    pipeline_config["data"]["dataset_id"] = "nonexisting"
    assert not sup.validate_system(pipeline_config)


@patch("modyn.supervisor.internal.supervisor.MetadataDatabaseConnection", MockDatabaseConnection)
def test_register_pipeline():
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter

    pipeline_config = get_minimal_pipeline_config()
    pipeline_id = sup.register_pipeline(pipeline_config)
    assert pipeline_id == 0


@patch("modyn.supervisor.internal.supervisor.MetadataDatabaseConnection", MockDatabaseConnection)
def test_unregister_pipeline():
    # TODO(#64,#124,#302): implement a real test when func is implemented.
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    sup.unregister_pipeline(PIPELINE_ID)


@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(GRPCHandler, "get_time_at_storage", return_value=START_TIMESTAMP)
@patch.object(Supervisor, "register_pipeline", return_value=PIPELINE_ID)
def test_start_pipeline(
    test_register_pipeline,
    test_get_time_at_storage,
    test_trainer_server_available,
    test_dataset_availabale,
) -> None:
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    pipeline_config = get_minimal_pipeline_config()

    mock_start = mock.Mock()
    mock_start.side_effect = noop
    with patch("multiprocessing.Process.start", mock_start):
        res = sup.start_pipeline(pipeline_config, EVALUATION_DIRECTORY)
        assert res["pipeline_id"] == PIPELINE_ID
        assert isinstance(sup._pipeline_process_dict[res["pipeline_id"]], PipelineInfo)


def test_start_pipeline_throws_on_invalid_pipeline_config() -> None:
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    res = sup.start_pipeline({}, EVALUATION_DIRECTORY)
    assert res["pipeline_id"] == -1
    assert "exception" in res and res["exception"] == "Invalid pipeline configuration"


@patch.object(GRPCHandler, "init_selector", return_value=None)
@patch.object(GRPCHandler, "init_storage", return_value=None)
@patch.object(GRPCHandler, "init_trainer_server", return_value=None)
@patch.object(GRPCHandler, "init_evaluator", return_value=None)
@patch("modyn.utils.grpc_connection_established", return_value=True)
@patch.object(GRPCHandler, "dataset_available", return_value=False)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(Supervisor, "init_metadata_db", noop_init_metadata_db)
def test_start_pipeline_throws_on_invalid_system_config(
    test_init_selector,
    test_init_storage,
    test_init_trainer_server,
    test_init_evaluator,
    test_connection_established,
    test_trainer_server_available,
    test_dataset_available,
) -> None:
    sup = Supervisor({})
    sup.init_cluster_connection()
    res = sup.start_pipeline(get_minimal_pipeline_config(), EVALUATION_DIRECTORY)
    assert res["pipeline_id"] == -1
    assert "exception" in res and res["exception"] == "Invalid system configuration"


@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(GRPCHandler, "get_time_at_storage", return_value=START_TIMESTAMP)
@patch.object(Supervisor, "register_pipeline", return_value=PIPELINE_ID)
@patch("modyn.supervisor.internal.pipeline_executor", "execute_pipeline", time.sleep(1))
@patch.object(PipelineInfo, "get_msg_from_queue", return_value=None)
def test_get_pipeline_status_running(
    test_get_msg_from_queue,
    test_register_pipeline,
    test_get_time_at_storage,
    test_trainer_server_available,
    test_dataset_availabale,
) -> None:
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    pipeline_config = get_minimal_pipeline_config()

    # mock_start = mock.Mock()
    # mock_start.side_effect = noop
    # with patch("multiprocessing.Process.start", mock_start):

    res = sup.start_pipeline(pipeline_config, EVALUATION_DIRECTORY)
    msg = sup.get_pipeline_status(res["pipeline_id"])
    assert msg["status"] == PipelineStatus.RUNNING


@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(GRPCHandler, "get_time_at_storage", return_value=START_TIMESTAMP)
@patch.object(Supervisor, "register_pipeline", return_value=PIPELINE_ID)
@patch("modyn.supervisor.internal.pipeline_executor", "execute_pipeline", noop)
@patch.object(PipelineInfo, "get_msg_from_queue", return_value=None)
def test_get_pipeline_status_exit(
    test_get_msg_from_queue,
    test_register_pipeline,
    test_get_time_at_storage,
    test_trainer_server_available,
    test_dataset_availabale,
) -> None:
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    pipeline_config = get_minimal_pipeline_config()

    # mock_start = mock.Mock()
    # mock_start.side_effect = noop
    # with patch("multiprocessing.Process.start", mock_start):

    start_time = time.time()
    res = sup.start_pipeline(pipeline_config, EVALUATION_DIRECTORY)

    # iteratively query the pipeline status until it is no longer running
    while time.time() - start_time < 30:
        msg = sup.get_pipeline_status(res["pipeline_id"])
        if msg["status"] != PipelineStatus.RUNNING:
            break
        time.sleep(2)
    else:
        raise TimeoutError("Pipeline did not finish in time")

    assert msg["status"] == PipelineStatus.EXIT


@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(GRPCHandler, "get_time_at_storage", return_value=START_TIMESTAMP)
@patch.object(Supervisor, "register_pipeline", return_value=PIPELINE_ID)
def test_get_pipeline_status_not_found(
    test_register_pipeline,
    test_get_time_at_storage,
    test_trainer_server_available,
    test_dataset_availabale,
) -> None:
    sup = get_non_connecting_supervisor()  # pylint: disable=no-value-for-parameter
    msg = sup.get_pipeline_status(PIPELINE_ID)
    assert msg["status"] == PipelineStatus.NOTFOUND
