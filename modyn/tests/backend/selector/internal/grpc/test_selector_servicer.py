# pylint: disable=unused-argument, no-name-in-module
# TODO(MaxiBoether): this tests multiple things, such as entrypoint, servicer and more.
# split this into entrypoint, server, servicer

import json
import os
import pathlib

import pytest
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501, E611
    DataInformRequest,
    GetSamplesRequest,
    JsonString,
    PipelineResponse,
    RegisterPipelineRequest,
)
from modyn.backend.selector.internal.grpc.selector_server import SelectorServer

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
        metadata = Metadata("test_key", 100, 0.5, False, 1, b"test_data", pipeline_id, 42)

        metadata.metadata_id = 1  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata)

        metadata2 = Metadata("test_key2", 101, 0.75, True, 2, b"test_data2", pipeline_id, 42)

        metadata2.metadata_id = 2  # SQLite does not support autoincrement for composite primary key
        database.session.add(metadata2)

        database.session.commit()


def teardown():
    os.remove(database_path)


def test_illegal_register_raises():
    selection_strategy = {
        "name": "finetune",
        "config": {"limit": 8, "reset_after_trigger": False, "unused_data_ratio": 50},
    }
    selector_server = SelectorServer(get_minimal_modyn_config())
    servicer = selector_server.grpc_servicer
    with pytest.raises(ValueError):
        servicer.register_pipeline(
            RegisterPipelineRequest(
                num_workers=-1, selection_strategy=JsonString(value=json.dumps(selection_strategy))
            ),
            None,
        )


def test_full_cycle():
    selection_strategy = {
        "name": "finetune",
        "config": {"limit": 8, "reset_after_trigger": False, "unused_data_ratio": 50},
    }

    selector_server = SelectorServer(get_minimal_modyn_config())
    servicer = selector_server.grpc_servicer
    pipeline_response: PipelineResponse = servicer.register_pipeline(
        RegisterPipelineRequest(num_workers=1, selection_strategy=JsonString(value=json.dumps(selection_strategy))),
        None,
    )
    pipeline_id = pipeline_response.pipeline_id
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

    assert set(worker_0_samples) == set(["test_key_1", "test_key_2", "test_key_3", "test_key_4"])

    # Now check that we correctly raise errors in the case of silly requests.

    with pytest.raises(ValueError):
        # Pipeline ID not registered
        servicer.get_sample_keys_and_weight(GetSamplesRequest(pipeline_id=3, trigger_id=trigger_id, worker_id=0), None)

    with pytest.raises(ValueError):
        # Num workers out of bounds
        servicer.get_sample_keys_and_weight(
            GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1), None
        )

    with pytest.raises(ValueError):
        # Num workers out of bounds
        servicer.get_sample_keys_and_weight(
            GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=-1), None
        )

    with pytest.raises(ValueError):
        # Pipeline ID not registered
        servicer.inform_data(
            DataInformRequest(pipeline_id=3, keys=data_keys_1, timestamps=data_timestamps_1, labels=data_labels_1),
            None,
        )

    with pytest.raises(ValueError):
        # Pipeline ID not registered
        servicer.inform_data_and_trigger(
            DataInformRequest(pipeline_id=3, keys=data_keys_1, timestamps=data_timestamps_1, labels=data_labels_1),
            None,
        )
