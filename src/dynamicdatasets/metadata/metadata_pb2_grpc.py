# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import dynamicdatasets.metadata.metadata_pb2 as metadata__pb2


class MetadataStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AddBatch = channel.unary_unary(
                '/metadata.Metadata/AddBatch',
                request_serializer=metadata__pb2.Batch.SerializeToString,
                response_deserializer=metadata__pb2.BatchId.FromString,
                )
        self.GetBatch = channel.unary_unary(
                '/metadata.Metadata/GetBatch',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=metadata__pb2.BatchData.FromString,
                )


class MetadataServicer(object):
    """Missing associated documentation comment in .proto file."""

    def AddBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetBatch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MetadataServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'AddBatch': grpc.unary_unary_rpc_method_handler(
                    servicer.AddBatch,
                    request_deserializer=metadata__pb2.Batch.FromString,
                    response_serializer=metadata__pb2.BatchId.SerializeToString,
            ),
            'GetBatch': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBatch,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=metadata__pb2.BatchData.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'metadata.Metadata', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Metadata(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def AddBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/AddBatch',
            metadata__pb2.Batch.SerializeToString,
            metadata__pb2.BatchId.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/GetBatch',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            metadata__pb2.BatchData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
