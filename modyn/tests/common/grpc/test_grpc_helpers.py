import contextlib
from typing import Callable

from modyn.common.grpc import GenericGRPCServer

# TODO(create issue): add more meaningful tests


def test_init():
    GenericGRPCServer({}, "1234", lambda x: None)
