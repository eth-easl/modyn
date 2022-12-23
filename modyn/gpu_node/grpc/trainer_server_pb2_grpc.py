# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import trainer_server_pb2 as trainer__server__pb2


class TrainerServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.start_training = channel.unary_unary(
                '/trainer.TrainerServer/start_training',
                request_serializer=trainer__server__pb2.TrainerServerRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.TrainerServerResponse.FromString,
                )
        self.trainer_available = channel.unary_unary(
                '/trainer.TrainerServer/trainer_available',
                request_serializer=trainer__server__pb2.TrainerAvailableRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.TrainerAvailableResponse.FromString,
                )


class TrainerServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def start_training(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def trainer_available(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainerServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'start_training': grpc.unary_unary_rpc_method_handler(
                    servicer.start_training,
                    request_deserializer=trainer__server__pb2.TrainerServerRequest.FromString,
                    response_serializer=trainer__server__pb2.TrainerServerResponse.SerializeToString,
            ),
            'trainer_available': grpc.unary_unary_rpc_method_handler(
                    servicer.trainer_available,
                    request_deserializer=trainer__server__pb2.TrainerAvailableRequest.FromString,
                    response_serializer=trainer__server__pb2.TrainerAvailableResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'trainer.TrainerServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TrainerServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def start_training(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trainer.TrainerServer/start_training',
            trainer__server__pb2.TrainerServerRequest.SerializeToString,
            trainer__server__pb2.TrainerServerResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def trainer_available(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trainer.TrainerServer/trainer_available',
            trainer__server__pb2.TrainerAvailableRequest.SerializeToString,
            trainer__server__pb2.TrainerAvailableResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
