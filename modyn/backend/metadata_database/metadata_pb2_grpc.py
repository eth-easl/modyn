# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import modyn.backend.metadata_database.metadata_pb2 as metadata__pb2

# pylint: disable-next=invalid-name

class MetadataStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetByKeys = channel.unary_unary(
                '/metadata.Metadata/GetByKeys',
                request_serializer=metadata__pb2.GetByKeysRequest.SerializeToString,
                response_deserializer=metadata__pb2.GetResponse.FromString,
                )
        self.GetByQuery = channel.unary_unary(
                '/metadata.Metadata/GetByQuery',
                request_serializer=metadata__pb2.GetByQueryRequest.SerializeToString,
                response_deserializer=metadata__pb2.GetResponse.FromString,
                )
        self.GetKeysByQuery = channel.unary_unary(
                '/metadata.Metadata/GetKeysByQuery',
                request_serializer=metadata__pb2.GetByQueryRequest.SerializeToString,
                response_deserializer=metadata__pb2.GetKeysResponse.FromString,
                )
        self.Set = channel.unary_unary(
                '/metadata.Metadata/Set',
                request_serializer=metadata__pb2.SetRequest.SerializeToString,
                response_deserializer=metadata__pb2.SetResponse.FromString,
                )
        self.DeleteTraining = channel.unary_unary(
                '/metadata.Metadata/DeleteTraining',
                request_serializer=metadata__pb2.DeleteRequest.SerializeToString,
                response_deserializer=metadata__pb2.DeleteResponse.FromString,
                )
        self.RegisterTraining = channel.unary_unary(
                '/metadata.Metadata/RegisterTraining',
                request_serializer=metadata__pb2.RegisterRequest.SerializeToString,
                response_deserializer=metadata__pb2.RegisterResponse.FromString,
                )
        self.GetTrainingInfo = channel.unary_unary(
                '/metadata.Metadata/GetTrainingInfo',
                request_serializer=metadata__pb2.GetTrainingRequest.SerializeToString,
                response_deserializer=metadata__pb2.TrainingResponse.FromString,
                )


class MetadataServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetByKeys(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetByQuery(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetKeysByQuery(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Set(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteTraining(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterTraining(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTrainingInfo(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MetadataServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetByKeys': grpc.unary_unary_rpc_method_handler(
                    servicer.GetByKeys,
                    request_deserializer=metadata__pb2.GetByKeysRequest.FromString,
                    response_serializer=metadata__pb2.GetResponse.SerializeToString,
            ),
            'GetByQuery': grpc.unary_unary_rpc_method_handler(
                    servicer.GetByQuery,
                    request_deserializer=metadata__pb2.GetByQueryRequest.FromString,
                    response_serializer=metadata__pb2.GetResponse.SerializeToString,
            ),
            'GetKeysByQuery': grpc.unary_unary_rpc_method_handler(
                    servicer.GetKeysByQuery,
                    request_deserializer=metadata__pb2.GetByQueryRequest.FromString,
                    response_serializer=metadata__pb2.GetKeysResponse.SerializeToString,
            ),
            'Set': grpc.unary_unary_rpc_method_handler(
                    servicer.Set,
                    request_deserializer=metadata__pb2.SetRequest.FromString,
                    response_serializer=metadata__pb2.SetResponse.SerializeToString,
            ),
            'DeleteTraining': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteTraining,
                    request_deserializer=metadata__pb2.DeleteRequest.FromString,
                    response_serializer=metadata__pb2.DeleteResponse.SerializeToString,
            ),
            'RegisterTraining': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterTraining,
                    request_deserializer=metadata__pb2.RegisterRequest.FromString,
                    response_serializer=metadata__pb2.RegisterResponse.SerializeToString,
            ),
            'GetTrainingInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTrainingInfo,
                    request_deserializer=metadata__pb2.GetTrainingRequest.FromString,
                    response_serializer=metadata__pb2.TrainingResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'metadata.Metadata', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Metadata(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetByKeys(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/GetByKeys',
            metadata__pb2.GetByKeysRequest.SerializeToString,
            metadata__pb2.GetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetByQuery(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/GetByQuery',
            metadata__pb2.GetByQueryRequest.SerializeToString,
            metadata__pb2.GetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetKeysByQuery(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/GetKeysByQuery',
            metadata__pb2.GetByQueryRequest.SerializeToString,
            metadata__pb2.GetKeysResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Set(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/Set',
            metadata__pb2.SetRequest.SerializeToString,
            metadata__pb2.SetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteTraining(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/DeleteTraining',
            metadata__pb2.DeleteRequest.SerializeToString,
            metadata__pb2.DeleteResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterTraining(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/RegisterTraining',
            metadata__pb2.RegisterRequest.SerializeToString,
            metadata__pb2.RegisterResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTrainingInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/metadata.Metadata/GetTrainingInfo',
            metadata__pb2.GetTrainingRequest.SerializeToString,
            metadata__pb2.TrainingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
