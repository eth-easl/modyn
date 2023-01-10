# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import metadata_processor_pb2 as metadata__processor__pb2


class MetadataProcessorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessPostTrainingMetadata = channel.unary_unary(
                '/metadata_processor.MetadataProcessor/ProcessPostTrainingMetadata',
                request_serializer=metadata__processor__pb2.PostTrainingMetadataRequest.SerializeToString,
                response_deserializer=metadata__processor__pb2.PostTrainingMetadataResponse.FromString,
                )


class MetadataProcessorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessPostTrainingMetadata(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MetadataProcessorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessPostTrainingMetadata': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessPostTrainingMetadata,
                    request_deserializer=metadata__processor__pb2.PostTrainingMetadataRequest.FromString,
                    response_serializer=metadata__processor__pb2.PostTrainingMetadataResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'metadata_processor.MetadataProcessor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MetadataProcessor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessPostTrainingMetadata(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata_processor.MetadataProcessor/ProcessPostTrainingMetadata',
            metadata__processor__pb2.PostTrainingMetadataRequest.SerializeToString,
            metadata__processor__pb2.PostTrainingMetadataResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
