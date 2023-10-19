# pylint: disable=unused-argument

from modyn.storage.internal.grpc.grpc_server import StorageGRPCServer


def get_modyn_config():
    return {"storage": {"port": "50051", "type": "grpc", "sample_batch_size": 1024}}


def test_init():
    grpc_server = StorageGRPCServer(get_modyn_config())
    assert grpc_server.modyn_config == get_modyn_config()
