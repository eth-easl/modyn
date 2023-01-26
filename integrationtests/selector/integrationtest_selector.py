from integrationtests.utils import get_modyn_config
from modyn.utils import grpc_connection_established

import grpc

def connect_to_selector_servicer() -> grpc.Channel:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        assert False, f"Could not establish gRPC connection to selector at {selector_address}."

    return selector_channel


def test_selector() -> None:
    pass


if __name__ == '__main__':
    test_selector()
