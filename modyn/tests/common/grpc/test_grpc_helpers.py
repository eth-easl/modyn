from modyn.common.grpc import GenericGRPCServer

# TODO(310): add more meaningful tests


def test_init():
    GenericGRPCServer({}, "1234", lambda x: None)
