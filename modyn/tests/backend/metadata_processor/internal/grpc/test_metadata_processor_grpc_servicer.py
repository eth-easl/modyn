# pylint: disable-next=no-name-in-module
from unittest.mock import MagicMock, patch

from modyn.backend.metadata_processor.internal.grpc.generated.metadata_processor_pb2 import (
    PerSampleMetadata,
    PerTriggerMetadata,
    PipelineResponse,
    RegisterPipelineRequest,
    TrainingMetadataRequest,
    TrainingMetadataResponse,
)
from modyn.backend.metadata_processor.internal.grpc.metadata_processor_grpc_servicer import (
    MetadataProcessorGRPCServicer,
)
from modyn.backend.metadata_processor.internal.metadata_processor_manager import MetadataProcessorManager

TRIGGER_METATDATA = PerTriggerMetadata(loss=0.05)
SAMPLE_METADATA = [PerSampleMetadata(sample_id="s1", loss=0.1)]


def test_constructor():
    manager = MetadataProcessorManager({})
    servicer = MetadataProcessorGRPCServicer(manager)
    assert servicer.processor_manager == manager


@patch.object(MetadataProcessorManager, "register_pipeline")
def test_register_pipeline(test__register_pipeline: MagicMock):
    manager = MetadataProcessorManager({})
    servicer = MetadataProcessorGRPCServicer(manager)
    request = RegisterPipelineRequest(pipeline_id=56, processor_type="proc")

    response = servicer.register_pipeline(request, None)

    assert isinstance(response, PipelineResponse)
    test__register_pipeline.assert_called_once_with(56, "proc")


@patch.object(MetadataProcessorManager, "process_training_metadata")
def test_process_training_metadata(test__process_training_metadata: MagicMock):
    manager = MetadataProcessorManager({})
    servicer = MetadataProcessorGRPCServicer(manager)
    request = TrainingMetadataRequest(
        pipeline_id=56, trigger_id=1, trigger_metadata=TRIGGER_METATDATA, sample_metadata=SAMPLE_METADATA
    )

    response = servicer.process_training_metadata(request, None)

    assert isinstance(response, TrainingMetadataResponse)
    test__process_training_metadata.assert_called_once_with(56, 1, request.trigger_metadata, request.sample_metadata)
