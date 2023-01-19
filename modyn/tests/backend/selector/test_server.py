# pylint: disable=unused-argument, no-name-in-module
import os
import pathlib
import sys
from unittest.mock import patch

import grpc
import pytest
import yaml
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import GetSamplesRequest  # noqa: E402, E501, E611
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

        trainig = Training(1, 1)
        database.get_session().add(trainig)
        database.get_session().commit()

        metadata = Metadata("test_key", 0.5, False, 1, b"test_data", trainig.id)

        metadata.id = 1  # SQLite does not support autoincrement for composite primary keys #pylint
        database.get_session().add(metadata)

        metadata2 = Metadata("test_key2", 0.75, True, 2, b"test_data2", trainig.id)

        metadata2.id = 2  # SQLite does not support autoincrement for composite primary key
        database.get_session().add(metadata2)

        database.get_session().commit()


def teardown():
    os.remove(database_path)


def test_prepare_training_set():
    with open("modyn/config/examples/example-pipeline.yaml", "r", encoding="utf-8") as pipeline_file:
        pipeline_cfg = yaml.safe_load(pipeline_file)

    selector_server = SelectorServer(pipeline_cfg, get_minimal_modyn_config())
    servicer = selector_server.grpc_server

    assert selector_server.selector.register_training(training_set_size=8, num_workers=1) == 2

    assert set(
        servicer.get_sample_keys_and_metadata(
            GetSamplesRequest(training_id=1, training_set_number=0, worker_id=0), None
        ).training_samples_subset
    ) == set(["test_key"])


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
    # testargs = ["selector_entrypoint.py", "modyn/config/config.yaml"]
    testargs = [
        "selector_entrypoint.py",
        "modyn/config/examples/modyn_config.yaml",
        "modyn/config/examples/example-pipeline.yaml",
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
