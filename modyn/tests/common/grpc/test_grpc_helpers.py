import contextlib
from typing import Callable

from modyn.common.grpc import GenericGRPCServer

# TODO(create issue): add more meaningful tests


@contextlib.contextmanager
def mock_context_mgr(port: str):
    yield port


def mock_run_server_worker(
    bind_address: str, add_servicer_callback: Callable, modyn_config: dict, callback_kwargs: dict
):
    pass


def mock_callback(arg1, arg2):
    pass


def mock__wait_forever():
    pass


def test_init():
    GenericGRPCServer({}, "1234", lambda x: None)
