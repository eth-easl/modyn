# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import warnings

import grpc
import modyn.metadata_processor.internal.grpc.generated.metadata_processor_pb2 as metadata__processor__pb2

GRPC_GENERATED_VERSION = "1.63.0"
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = "1.65.0"
SCHEDULED_RELEASE_DATE = "June 25, 2024"
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + f" but the generated code in metadata_processor_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
        + f" This warning will become an error in {EXPECTED_ERROR_RELEASE},"
        + f" scheduled for release on {SCHEDULED_RELEASE_DATE}.",
        RuntimeWarning,
    )


class MetadataProcessorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.register_pipeline = channel.unary_unary(
            "/metadata_processor.MetadataProcessor/register_pipeline",
            request_serializer=metadata__processor__pb2.RegisterPipelineRequest.SerializeToString,
            response_deserializer=metadata__processor__pb2.PipelineResponse.FromString,
            _registered_method=True,
        )
        self.process_training_metadata = channel.unary_unary(
            "/metadata_processor.MetadataProcessor/process_training_metadata",
            request_serializer=metadata__processor__pb2.TrainingMetadataRequest.SerializeToString,
            response_deserializer=metadata__processor__pb2.TrainingMetadataResponse.FromString,
            _registered_method=True,
        )


class MetadataProcessorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def register_pipeline(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def process_training_metadata(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_MetadataProcessorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "register_pipeline": grpc.unary_unary_rpc_method_handler(
            servicer.register_pipeline,
            request_deserializer=metadata__processor__pb2.RegisterPipelineRequest.FromString,
            response_serializer=metadata__processor__pb2.PipelineResponse.SerializeToString,
        ),
        "process_training_metadata": grpc.unary_unary_rpc_method_handler(
            servicer.process_training_metadata,
            request_deserializer=metadata__processor__pb2.TrainingMetadataRequest.FromString,
            response_serializer=metadata__processor__pb2.TrainingMetadataResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler("metadata_processor.MetadataProcessor", rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class MetadataProcessor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def register_pipeline(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/metadata_processor.MetadataProcessor/register_pipeline",
            metadata__processor__pb2.RegisterPipelineRequest.SerializeToString,
            metadata__processor__pb2.PipelineResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def process_training_metadata(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/metadata_processor.MetadataProcessor/process_training_metadata",
            metadata__processor__pb2.TrainingMetadataRequest.SerializeToString,
            metadata__processor__pb2.TrainingMetadataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
