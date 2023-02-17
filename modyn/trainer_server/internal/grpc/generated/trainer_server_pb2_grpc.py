# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 as trainer__server__pb2


class TrainerServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.trainer_available = channel.unary_unary(
                '/trainer.TrainerServer/trainer_available',
                request_serializer=trainer__server__pb2.TrainerAvailableRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.TrainerAvailableResponse.FromString,
                )
        self.start_training = channel.unary_unary(
                '/trainer.TrainerServer/start_training',
                request_serializer=trainer__server__pb2.StartTrainingRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.StartTrainingResponse.FromString,
                )
        self.get_training_status = channel.unary_unary(
                '/trainer.TrainerServer/get_training_status',
                request_serializer=trainer__server__pb2.TrainingStatusRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.TrainingStatusResponse.FromString,
                )
        self.get_final_model = channel.unary_unary(
                '/trainer.TrainerServer/get_final_model',
                request_serializer=trainer__server__pb2.GetFinalModelRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.GetFinalModelResponse.FromString,
                )
        self.get_latest_model = channel.unary_unary(
                '/trainer.TrainerServer/get_latest_model',
                request_serializer=trainer__server__pb2.GetLatestModelRequest.SerializeToString,
                response_deserializer=trainer__server__pb2.GetLatestModelResponse.FromString,
                )


class TrainerServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def trainer_available(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def start_training(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_training_status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_final_model(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_latest_model(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainerServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'trainer_available': grpc.unary_unary_rpc_method_handler(
                    servicer.trainer_available,
                    request_deserializer=trainer__server__pb2.TrainerAvailableRequest.FromString,
                    response_serializer=trainer__server__pb2.TrainerAvailableResponse.SerializeToString,
            ),
            'start_training': grpc.unary_unary_rpc_method_handler(
                    servicer.start_training,
                    request_deserializer=trainer__server__pb2.StartTrainingRequest.FromString,
                    response_serializer=trainer__server__pb2.StartTrainingResponse.SerializeToString,
            ),
            'get_training_status': grpc.unary_unary_rpc_method_handler(
                    servicer.get_training_status,
                    request_deserializer=trainer__server__pb2.TrainingStatusRequest.FromString,
                    response_serializer=trainer__server__pb2.TrainingStatusResponse.SerializeToString,
            ),
            'get_final_model': grpc.unary_unary_rpc_method_handler(
                    servicer.get_final_model,
                    request_deserializer=trainer__server__pb2.GetFinalModelRequest.FromString,
                    response_serializer=trainer__server__pb2.GetFinalModelResponse.SerializeToString,
            ),
            'get_latest_model': grpc.unary_unary_rpc_method_handler(
                    servicer.get_latest_model,
                    request_deserializer=trainer__server__pb2.GetLatestModelRequest.FromString,
                    response_serializer=trainer__server__pb2.GetLatestModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'trainer.TrainerServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TrainerServer(object):
    """Missing associated documentation comment in .proto file."""

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
            trainer__server__pb2.StartTrainingRequest.SerializeToString,
            trainer__server__pb2.StartTrainingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_training_status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trainer.TrainerServer/get_training_status',
            trainer__server__pb2.TrainingStatusRequest.SerializeToString,
            trainer__server__pb2.TrainingStatusResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_final_model(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trainer.TrainerServer/get_final_model',
            trainer__server__pb2.GetFinalModelRequest.SerializeToString,
            trainer__server__pb2.GetFinalModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_latest_model(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/trainer.TrainerServer/get_latest_model',
            trainer__server__pb2.GetLatestModelRequest.SerializeToString,
            trainer__server__pb2.GetLatestModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
