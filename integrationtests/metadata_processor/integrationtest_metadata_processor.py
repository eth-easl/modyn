import json
import os
import pathlib
from typing import Iterable

import grpc
import yaml
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import GetByKeysRequest, GetResponse
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    TrainingMetadataRequest,
    TrainingMetadataResponse,
)
from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import MetadataProcessorStub
from modyn.utils import grpc_connection_established

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"


class MetadataProcessorClient:
    def __init__(self, metadata_processor_channel: grpc.Channel) -> None:
        self._stub = MetadataProcessorStub(metadata_processor_channel)

    def send_metadata(self, training_id: int, data: str) -> TrainingMetadataResponse:
        req = TrainingMetadataRequest(training_id=training_id, data=data)
        resp = self._stub.ProcessTrainingMetadata(req)
        return resp


class MetadataDatabaseClient:
    def __init__(self, metadata_database_channel: grpc.Channel) -> None:
        self._stub = MetadataStub(grpc.insecure_channel())

    def get_metadata(self, training_id: int, keys: Iterable[str]) -> GetResponse:
        req = GetByKeysRequest(training_id=training_id, keys=keys)
        resp = self._stub.GetByKeys(req)
        return resp


def get_modyn_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def get_grpc_channel(config: dict, component: str) -> grpc.Channel:
    address = f"{config[component]['hostname']}:{config[component]['port']}"
    channel = grpc.insecure_channel(address)

    if not grpc_connection_established(channel):
        assert False, f"Could not establish gRPC connection to {component} at {address}"

    return channel


def send_metadata_and_check_database(
    processor_client: MetadataProcessorClient, database_client: MetadataDatabaseClient
) -> None:
    training_id = 10
    data = {
        "sample1": "metadata01",
        "sample2": "metadata02",
        "sample3": "metadata03"
    }
    serialized_data = json.loads(data)

    resp = processor_client.send_metadata(training_id, serialized_data)
    assert resp, "Coult not send training metadata to the Metadata Processor Server"

    database_resp = database_client.get_metadata(training_id, data.keys())
    assert database_resp, "Could not get metadata from the Metadata Database Server"

    assert len(database_resp.keys) == len(data), (
        f"Metadata database sent back {len(database_resp.keys)} keys, expected {len(data)}")
    assert len(database_resp.data) == len(data), (
        f"Metadata database sent back {len(database_resp.data)} metadata values, expected {len(data)}")
    assert len(database_resp.seen) == len(data), (
        f"Metadata database sent back {len(database_resp.seen)} seen flags, expected {len(data)}")

    assert database_resp.keys == list(data.keys()), (
        f"Metadata database sent back keys: {str(database_resp.keys)}, expected: {str(list(data.keys()))}")
    assert database_resp.seen == [True] * len(data), (
        f"Metadata database sent back seen flags: {str(database_resp.seen)}, expected all values to be True")
    assert database_resp.data == list(data.values()), (
        f"Metadata database sent back metadata values: {str(database_resp.data)}, expected: {str(list(data.values()))}")


def test_metadata_processor() -> None:
    config = get_modyn_config()

    processor_channel = get_grpc_channel(config, "metadata_processor")
    database_channel = get_grpc_channel(config, "metadata_database")

    processor_client = MetadataProcessorClient(processor_channel)
    database_client = MetadataDatabaseClient(database_channel)

    send_metadata_and_check_database(processor_client, database_client)


if __name__ == "__main__":
    test_metadata_processor()
