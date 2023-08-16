import os
import pathlib
from math import isclose

import grpc
import yaml
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SampleTrainingMetadata, TriggerTrainingMetadata
from modyn.metadata_database.utils import ModelStorageStrategyConfig

# pylint: disable-next=no-name-in-module
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (  # noqa: E402, E501
    PerSampleMetadata,
    PerTriggerMetadata,
    TrainingMetadataRequest,
    TrainingMetadataResponse,
)
from modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2_grpc import MetadataProcessorStub
from modyn.utils import grpc_connection_established

SCRIPT_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_FILE = SCRIPT_PATH.parent.parent.parent / "modyn" / "config" / "examples" / "modyn_config.yaml"


class MetadataProcessorClient:
    def __init__(self, metadata_processor_channel: grpc.Channel) -> None:
        self._stub = MetadataProcessorStub(metadata_processor_channel)

    def send_metadata(self, req: TrainingMetadataRequest) -> TrainingMetadataResponse:
        resp = self._stub.ProcessTrainingMetadata(req)
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


def send_metadata_and_check_database(processor_client: MetadataProcessorClient, config: dict) -> int:
    with MetadataDatabaseConnection(config) as database:
        pipeline_id = database.register_pipeline(
            2, "ResNet18", "{}", False, ModelStorageStrategyConfig("PyTorchFullModel")
        )

    req = TrainingMetadataRequest(
        pipeline_id=pipeline_id,
        trigger_id=1,
        trigger_metadata=PerTriggerMetadata(loss=0.5),
        sample_metadata=[PerSampleMetadata(sample_id="s1", loss=0.1), PerSampleMetadata(sample_id="s2", loss=0.2)],
    )

    resp = processor_client.send_metadata(req)
    assert resp, "Coult not send training metadata to the Metadata Processor Server"

    with MetadataDatabaseConnection(config) as database:
        trigger_metadata = (
            database.session.query(
                TriggerTrainingMetadata.trigger_id,
                TriggerTrainingMetadata.pipeline_id,
                TriggerTrainingMetadata.overall_loss,
            )
            .filter(TriggerTrainingMetadata.pipeline_id == pipeline_id)
            .all()
        )

        tids, pids, overall_loss = zip(*trigger_metadata)

        assert len(trigger_metadata) == 1, f"Expected 1 entry for trigger metadata in db, found {len(trigger_metadata)}"
        assert tids[0] == 1, f"Expected trigger ID 1 in db, found {tids[0]}"
        assert isclose(overall_loss[0], 0.5), f"Expected overall loss 0.5 in db, found {overall_loss[0]}"

        sample_metadata = (
            database.session.query(
                SampleTrainingMetadata.pipeline_id,
                SampleTrainingMetadata.trigger_id,
                SampleTrainingMetadata.sample_key,
                SampleTrainingMetadata.loss,
            )
            .filter(SampleTrainingMetadata.pipeline_id == pipeline_id)
            .all()
        )

        pids, tids, keys, loss = zip(*sample_metadata)

        assert len(sample_metadata) == 2, f"Expected 2 entries for sample metadata in db, found {len(sample_metadata)}"
        assert (
            pids[0] == pipeline_id and pids[1] == pipeline_id
        ), f"Expected all sample metadata in db to be for pipeline ID 1, found {pids}"
        assert tids[0] == 1 and tids[1] == 1, f"Expected all sample metadata in db to be for trigger ID 1, found {tids}"
        assert keys == ("s1", "s2"), f"Expected sample keys (s1, s2) in db, found {keys}"
        assert isclose(loss[0], 0.1, rel_tol=1e-5), f"Expected sample loss 0.1, found {loss[0]}"
        assert isclose(loss[1], 0.2, rel_tol=1e-5), f"Expected sample loss 0.2, found {loss[1]}"
        return pipeline_id


def clear_database(config: dict, pipeline_id: int):
    with MetadataDatabaseConnection(config) as database:
        database.session.query(TriggerTrainingMetadata).filter(
            TriggerTrainingMetadata.pipeline_id == pipeline_id
        ).delete()
        database.session.query(SampleTrainingMetadata).filter(
            SampleTrainingMetadata.pipeline_id == pipeline_id
        ).delete()
        database.session.commit()


def test_metadata_processor() -> None:
    config = get_modyn_config()

    processor_channel = get_grpc_channel(config, "metadata_processor")
    processor_client = MetadataProcessorClient(processor_channel)

    pipeline_id = send_metadata_and_check_database(processor_client, config)

    clear_database(config, pipeline_id)


if __name__ == "__main__":
    test_metadata_processor()
