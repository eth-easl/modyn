# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from storage import storage_pb2 as storage__pb2


class StorageStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Get = channel.unary_unary(
            '/storage.Storage/Get',
            request_serializer=storage__pb2.GetRequest.SerializeToString,
            response_deserializer=storage__pb2.GetResponse.FromString,
        )
        self.Query = channel.unary_unary(
            '/storage.Storage/Query',
            request_serializer=storage__pb2.QueryRequest.SerializeToString,
            response_deserializer=storage__pb2.QueryResponse.FromString,
        )
        self.Put = channel.unary_unary(
            '/storage.Storage/Put',
            request_serializer=storage__pb2.PutRequest.SerializeToString,
            response_deserializer=storage__pb2.PutResponse.FromString,
        )


class StorageServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Get(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Query(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Put(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_StorageServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Get': grpc.unary_unary_rpc_method_handler(
            servicer.Get,
            request_deserializer=storage__pb2.GetRequest.FromString,
            response_serializer=storage__pb2.GetResponse.SerializeToString,
        ),
        'Query': grpc.unary_unary_rpc_method_handler(
            servicer.Query,
            request_deserializer=storage__pb2.QueryRequest.FromString,
            response_serializer=storage__pb2.QueryResponse.SerializeToString,
        ),
        'Put': grpc.unary_unary_rpc_method_handler(
            servicer.Put,
            request_deserializer=storage__pb2.PutRequest.FromString,
            response_serializer=storage__pb2.PutResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'storage.Storage', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


class Storage(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Get(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/storage.Storage/Get',
            storage__pb2.GetRequest.SerializeToString,
            storage__pb2.GetResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata)

    @staticmethod
    def Query(request,
              target,
              options=(),
              channel_credentials=None,
              call_credentials=None,
              insecure=False,
              compression=None,
              wait_for_ready=None,
              timeout=None,
              metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/storage.Storage/Query',
            storage__pb2.QueryRequest.SerializeToString,
            storage__pb2.QueryResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata)

    @staticmethod
    def Put(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/storage.Storage/Put',
            storage__pb2.PutRequest.SerializeToString,
            storage__pb2.PutResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata)
