# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import modyn.storage.internal.grpc.generated.storage_pb2 as storage__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


class StorageStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Get = channel.unary_stream(
                '/modyn.storage.Storage/Get',
                request_serializer=storage__pb2.GetRequest.SerializeToString,
                response_deserializer=storage__pb2.GetResponse.FromString,
                )
        self.GetNewDataSince = channel.unary_stream(
                '/modyn.storage.Storage/GetNewDataSince',
                request_serializer=storage__pb2.GetNewDataSinceRequest.SerializeToString,
                response_deserializer=storage__pb2.GetNewDataSinceResponse.FromString,
                )
        self.GetDataInInterval = channel.unary_stream(
                '/modyn.storage.Storage/GetDataInInterval',
                request_serializer=storage__pb2.GetDataInIntervalRequest.SerializeToString,
                response_deserializer=storage__pb2.GetDataInIntervalResponse.FromString,
                )
        self.CheckAvailability = channel.unary_unary(
                '/modyn.storage.Storage/CheckAvailability',
                request_serializer=storage__pb2.DatasetAvailableRequest.SerializeToString,
                response_deserializer=storage__pb2.DatasetAvailableResponse.FromString,
                )
        self.RegisterNewDataset = channel.unary_unary(
                '/modyn.storage.Storage/RegisterNewDataset',
                request_serializer=storage__pb2.RegisterNewDatasetRequest.SerializeToString,
                response_deserializer=storage__pb2.RegisterNewDatasetResponse.FromString,
                )
        self.GetCurrentTimestamp = channel.unary_unary(
                '/modyn.storage.Storage/GetCurrentTimestamp',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=storage__pb2.GetCurrentTimestampResponse.FromString,
                )
        self.DeleteDataset = channel.unary_unary(
                '/modyn.storage.Storage/DeleteDataset',
                request_serializer=storage__pb2.DatasetAvailableRequest.SerializeToString,
                response_deserializer=storage__pb2.DeleteDatasetResponse.FromString,
                )


class StorageServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Get(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNewDataSince(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDataInInterval(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckAvailability(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterNewDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetCurrentTimestamp(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_StorageServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Get': grpc.unary_stream_rpc_method_handler(
                    servicer.Get,
                    request_deserializer=storage__pb2.GetRequest.FromString,
                    response_serializer=storage__pb2.GetResponse.SerializeToString,
            ),
            'GetNewDataSince': grpc.unary_stream_rpc_method_handler(
                    servicer.GetNewDataSince,
                    request_deserializer=storage__pb2.GetNewDataSinceRequest.FromString,
                    response_serializer=storage__pb2.GetNewDataSinceResponse.SerializeToString,
            ),
            'GetDataInInterval': grpc.unary_stream_rpc_method_handler(
                    servicer.GetDataInInterval,
                    request_deserializer=storage__pb2.GetDataInIntervalRequest.FromString,
                    response_serializer=storage__pb2.GetDataInIntervalResponse.SerializeToString,
            ),
            'CheckAvailability': grpc.unary_unary_rpc_method_handler(
                    servicer.CheckAvailability,
                    request_deserializer=storage__pb2.DatasetAvailableRequest.FromString,
                    response_serializer=storage__pb2.DatasetAvailableResponse.SerializeToString,
            ),
            'RegisterNewDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterNewDataset,
                    request_deserializer=storage__pb2.RegisterNewDatasetRequest.FromString,
                    response_serializer=storage__pb2.RegisterNewDatasetResponse.SerializeToString,
            ),
            'GetCurrentTimestamp': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCurrentTimestamp,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=storage__pb2.GetCurrentTimestampResponse.SerializeToString,
            ),
            'DeleteDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDataset,
                    request_deserializer=storage__pb2.DatasetAvailableRequest.FromString,
                    response_serializer=storage__pb2.DeleteDatasetResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'modyn.storage.Storage', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
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
        return grpc.experimental.unary_stream(request, target, '/modyn.storage.Storage/Get',
            storage__pb2.GetRequest.SerializeToString,
            storage__pb2.GetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNewDataSince(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/modyn.storage.Storage/GetNewDataSince',
            storage__pb2.GetNewDataSinceRequest.SerializeToString,
            storage__pb2.GetNewDataSinceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDataInInterval(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/modyn.storage.Storage/GetDataInInterval',
            storage__pb2.GetDataInIntervalRequest.SerializeToString,
            storage__pb2.GetDataInIntervalResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckAvailability(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modyn.storage.Storage/CheckAvailability',
            storage__pb2.DatasetAvailableRequest.SerializeToString,
            storage__pb2.DatasetAvailableResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterNewDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modyn.storage.Storage/RegisterNewDataset',
            storage__pb2.RegisterNewDatasetRequest.SerializeToString,
            storage__pb2.RegisterNewDatasetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetCurrentTimestamp(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modyn.storage.Storage/GetCurrentTimestamp',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            storage__pb2.GetCurrentTimestampResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modyn.storage.Storage/DeleteDataset',
            storage__pb2.DatasetAvailableRequest.SerializeToString,
            storage__pb2.DeleteDatasetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
