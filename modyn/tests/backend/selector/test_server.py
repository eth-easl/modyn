# pylint: disable=unused-argument, no-name-in-module
import json
import os
import pathlib
import sys
from unittest.mock import patch

import grpc
import pytest
import yaml
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501, E611
    DataInformRequest,
    GetSamplesRequest,
)
from modyn.backend.selector.selector_entrypoint import main
from modyn.backend.selector.selector_server import SelectorServer

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
    }


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    self._modyn_config = get_minimal_modyn_config()


def setup():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()


def populate_metadata_database(pipeline_id):
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        metadata = Metadata("test_key", 100, 0.5, False, 1, b"test_data", pipeline_id)

        metadata.metadata_id = 1  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata)

        metadata2 = Metadata("test_key2", 101, 0.75, True, 2, b"test_data2", pipeline_id)

        metadata2.metadata_id = 2  # SQLite does not support autoincrement for composite primary key
        database.session.add(metadata2)

        database.session.commit()


def teardown():
    os.remove(database_path)


def test_prepare_training_set():
    selector_strategy_configs = {"name": "finetune", "configs": {"limit": 8}}
    selector_server = SelectorServer(get_minimal_modyn_config())
    servicer = selector_server.grpc_server
    pipeline_id = servicer.selector_manager.register_pipeline(
        num_workers=1, strategy_configs=json.dumps(selector_strategy_configs)
    )
    servicer.selector_manager._selectors[pipeline_id]._strategy.training_set_size_limit = 8
    populate_metadata_database(pipeline_id)

    assert set(
        servicer.get_sample_keys_and_weight(
            GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=0, worker_id=0), None
        ).training_samples_subset
    ) == set(["test_key"])

    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.delete_training(pipeline_id)


def test_full_cycle():
    selector_strategy_configs = {"name": "finetune", "configs": {"limit": 8}}

    selector_server = SelectorServer(get_minimal_modyn_config())
    servicer = selector_server.grpc_server
    pipeline_id = servicer.selector_manager.register_pipeline(
        num_workers=1, strategy_configs=json.dumps(selector_strategy_configs)
    )
    servicer.selector_manager._selectors[pipeline_id]._strategy.training_set_size_limit = 8

    data_keys_1 = ["test_key_1", "test_key_2"]
    data_timestamps_1 = [0, 1]
    data_labels_1 = [0, 0]
    data_keys_2 = ["test_key_3", "test_key_4"]
    data_timestamps_2 = [2, 3]
    data_labels_2 = [1, 1]
    servicer.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id, keys=data_keys_1, timestamps=data_timestamps_1, labels=data_labels_1
        ),
        None,
    )
    trigger_response = servicer.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id, keys=data_keys_2, timestamps=data_timestamps_2, labels=data_labels_2
        ),
        None,
    )
    trigger_id = trigger_response.trigger_id

    worker_0_samples = servicer.get_sample_keys_and_weight(
        GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0), None
    ).training_samples_subset

    print(worker_0_samples)

    assert set(worker_0_samples) == set(["test_key_1", "test_key_2", "test_key_3", "test_key_4"])


class DummyServer:
    def __init__(self, arg):
        pass

    def add_insecure_port(self, arg=None):
        pass

    def start(self):
        pass

    def wait_for_termination(self):
        pass

    def add_generic_rpc_handlers(self, arg=None):
        pass


@patch.object(grpc, "server", return_value=DummyServer(None))
def test_main(test_server_mock):
    testargs = [
        "selector_entrypoint.py",
        "modyn/config/examples/modyn_config.yaml",
    ]
    with patch.object(sys, "argv", testargs):
        main()


def test_main_raise():
    testargs = [
        "selector_entrypoint.py",
        "modyn/config/examples/example-pipeline.yaml",
        "modyn/config/config.yaml",
        "extra",
    ]
    with patch.object(sys, "argv", testargs):
        with pytest.raises(SystemExit):
            main()
