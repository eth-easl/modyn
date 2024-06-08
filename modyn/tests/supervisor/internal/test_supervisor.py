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

from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.system import ModynConfig, SupervisorConfig
from modyn.metadata_database.utils import ModelStorageStrategyConfig
from modyn.supervisor.internal.eval.result_writer import JsonResultWriter, TensorboardResultWriter
from modyn.supervisor.internal.grpc.enums import PipelineStatus
from modyn.supervisor.internal.grpc_handler import GRPCHandler
from modyn.supervisor.internal.supervisor import Supervisor
from modyn.supervisor.internal.utils import PipelineInfo

EVALUATION_DIRECTORY: pathlib.Path = pathlib.Path(os.path.realpath(__file__)).parent / "test_eval_dir"
SUPPORTED_EVAL_RESULT_WRITERS: dict = {
    "json": JsonResultWriter,
    "tensorboard": TensorboardResultWriter,
}
START_TIMESTAMP = 21
PIPELINE_ID = 42


@pytest.fixture
def minimal_system_config(dummy_system_config: ModynConfig) -> ModynConfig:
    config = dummy_system_config.model_copy()
    config.supervisor = SupervisorConfig(hostname="localhost", port=50051, eval_directory=EVALUATION_DIRECTORY)
    return config


def noop_init_metadata_db(self) -> None:
    pass


def noop_pipeline_executor_constructor_mock(
    self,
    start_timestamp: int,
    pipeline_id: int,
    modyn_config: dict,
    pipeline_config: dict,
    eval_directory: str,
    pipeline_status_queue: mp.Queue,
    training_status_queue: mp.Queue,
    eval_status_queue: mp.Queue,
    start_replay_at: Optional[int] = None,
    stop_replay_at: Optional[int] = None,
    maximum_triggers: Optional[int] = None,
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
        data_config: str,
        full_model_strategy: ModelStorageStrategyConfig,
        incremental_model_strategy: Optional[ModelStorageStrategyConfig] = None,
        full_model_interval: Optional[int] = None,
        auxiliary_pipeline_id: Optional[int] = None,
    ) -> int:
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
    minimal_system_config: ModynConfig,
    test_connection_established,
    test_init_evaluator,
    test_init_trainer_server,
    test_init_storage,
    test_init_selector,
) -> Supervisor:
    supervisor = Supervisor(minimal_system_config)
    supervisor.init_cluster_connection()

    return supervisor


def test_initialization(minimal_system_config: ModynConfig) -> None:
    get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter


def test_validate_pipeline_config_content(dummy_pipeline_config: ModynPipelineConfig):
    # Check that our minimal pipeline config gets accepted
    assert Supervisor.validate_pipeline_config_content(dummy_pipeline_config)

    # Check that an unknown model gets rejected
    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_config.modyn_model.id = "UnknownModel"
    assert not Supervisor.validate_pipeline_config_content(pipeline_config)

    # Check that an unknown trigger gets rejected
    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_config.trigger.id = "UnknownTrigger"
    assert not Supervisor.validate_pipeline_config_content(pipeline_config)


def test_validate_pipeline_config(dummy_pipeline_config: ModynPipelineConfig):
    # Check that our minimal pipeline config gets accepted
    pipeline_config = dummy_pipeline_config.model_copy()
    assert Supervisor.validate_pipeline_config(pipeline_config)

    # Check that an unknown model gets rejected
    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_config.modyn_model.id = "UnknownModel"
    assert not Supervisor.validate_pipeline_config(pipeline_config)


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
def test_dataset_available(minimal_system_config: ModynConfig, dummy_pipeline_config: ModynPipelineConfig):
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter

    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_config.data.dataset_id = "existing"
    assert sup.dataset_available(pipeline_config)

    pipeline_config.data.dataset_id = "nonexisting"
    assert not sup.dataset_available(pipeline_config)


@patch.object(GRPCHandler, "dataset_available", lambda self, did: did == "existing")
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
def test_validate_system(
    test_trainer_server_available, minimal_system_config: ModynConfig, dummy_pipeline_config: ModynPipelineConfig
):
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter

    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_config.data.dataset_id = "existing"
    assert sup.validate_system(pipeline_config)

    pipeline_config.data.dataset_id = "nonexisting"
    assert not sup.validate_system(pipeline_config)


@patch(
    "modyn.supervisor.internal.supervisor.MetadataDatabaseConnection",
    MockDatabaseConnection,
)
def test_register_pipeline(minimal_system_config: ModynConfig, dummy_pipeline_config: ModynPipelineConfig):
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter

    pipeline_config = dummy_pipeline_config.model_copy()
    pipeline_id = sup.register_pipeline(pipeline_config)
    assert pipeline_id == 0


@patch(
    "modyn.supervisor.internal.supervisor.MetadataDatabaseConnection",
    MockDatabaseConnection,
)
def test_unregister_pipeline(minimal_system_config: ModynConfig):
    # TODO(#64,#124,#302): implement a real test when func is implemented.
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
    sup.unregister_pipeline(PIPELINE_ID)


@patch.object(GRPCHandler, "dataset_available", return_value=True)
@patch.object(GRPCHandler, "trainer_server_available", return_value=True)
@patch.object(GRPCHandler, "get_time_at_storage", return_value=START_TIMESTAMP)
@patch.object(Supervisor, "register_pipeline", return_value=PIPELINE_ID)
def test_start_pipeline(
    test_register_pipeline,
    test_get_time_at_storage,
    test_trainer_server_available,
    test_dataset_available,
    minimal_system_config: ModynConfig,
    dummy_pipeline_config: ModynPipelineConfig,
) -> None:
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
    pipeline_config = dummy_pipeline_config.model_copy()

    mock_start = mock.Mock()
    mock_start.side_effect = noop
    with patch("multiprocessing.Process.start", mock_start):
        res = sup.start_pipeline(pipeline_config, EVALUATION_DIRECTORY)
        assert res["pipeline_id"] == PIPELINE_ID
        assert isinstance(sup._pipeline_process_dict[res["pipeline_id"]], PipelineInfo)


def test_start_pipeline_throws_on_invalid_pipeline_config(minimal_system_config: ModynConfig) -> None:
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
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
    dummy_pipeline_config: ModynPipelineConfig,
    minimal_system_config: ModynConfig,
) -> None:
    sup = Supervisor(minimal_system_config)
    sup.init_cluster_connection()
    res = sup.start_pipeline(dummy_pipeline_config.model_copy(), EVALUATION_DIRECTORY)
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
    test_dataset_available,
    minimal_system_config: ModynConfig,
    dummy_pipeline_config: ModynPipelineConfig,
) -> None:
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
    pipeline_config = dummy_pipeline_config.model_copy()

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
    test_dataset_available,
    minimal_system_config: ModynConfig,
    dummy_pipeline_config: ModynPipelineConfig,
) -> None:
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
    pipeline_config = dummy_pipeline_config.model_copy()

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
    test_dataset_available,
    minimal_system_config: ModynConfig,
) -> None:
    sup = get_non_connecting_supervisor(minimal_system_config)  # pylint: disable=no-value-for-parameter
    msg = sup.get_pipeline_status(PIPELINE_ID)
    assert msg["status"] == PipelineStatus.NOTFOUND
