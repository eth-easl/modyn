# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import modyn.backend.selector.selector_pb2 as selector__pb2


class SelectorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.register_training = channel.unary_unary(
            '/selector.Selector/register_training',
            request_serializer=selector__pb2.RegisterTrainingRequest.SerializeToString,
            response_deserializer=selector__pb2.TrainingResponse.FromString,
        )
        self.get_sample_keys = channel.unary_unary(
            '/selector.Selector/get_sample_keys',
            request_serializer=selector__pb2.GetSamplesRequest.SerializeToString,
            response_deserializer=selector__pb2.SamplesResponse.FromString,
        )


class SelectorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def register_training(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_sample_keys(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SelectorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'register_training': grpc.unary_unary_rpc_method_handler(
            servicer.register_training,
            request_deserializer=selector__pb2.RegisterTrainingRequest.FromString,
            response_serializer=selector__pb2.TrainingResponse.SerializeToString,
        ),
        'get_sample_keys': grpc.unary_unary_rpc_method_handler(
            servicer.get_sample_keys,
            request_deserializer=selector__pb2.GetSamplesRequest.FromString,
            response_serializer=selector__pb2.SamplesResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'selector.Selector', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

 # This class is part of an EXPERIMENTAL API.


class Selector(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def register_training(request,
                          target,
                          options=(),
                          channel_credentials=None,
                          call_credentials=None,
                          insecure=False,
                          compression=None,
                          wait_for_ready=None,
                          timeout=None,
                          metadata=None):
        return grpc.experimental.unary_unary(request, target, '/selector.Selector/register_training',
                                             selector__pb2.RegisterTrainingRequest.SerializeToString,
                                             selector__pb2.TrainingResponse.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_sample_keys(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/selector.Selector/get_sample_keys',
                                             selector__pb2.GetSamplesRequest.SerializeToString,
                                             selector__pb2.SamplesResponse.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
