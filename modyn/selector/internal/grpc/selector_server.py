import atexit
import logging
import os
from typing import Any

from modyn.common.grpc import GenericGRPCServer
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import add_SelectorServicer_to_server  # noqa: E402, E501
from modyn.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.selector.internal.selector_manager import SelectorManager

logger = logging.getLogger(__name__)


class SelectorGRPCServer(GenericGRPCServer):
    @staticmethod
    def callback(modyn_config: dict, server: Any, selector_manager: SelectorManager) -> None:
        add_SelectorServicer_to_server(
            SelectorGRPCServicer(selector_manager, modyn_config["selector"]["sample_batch_size"]), server
        )

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.selector_manager = SelectorManager(modyn_config)

        callback_kwargs = {"selector_manager": self.selector_manager}
        super().__init__(modyn_config, modyn_config["selector"]["port"], SelectorGRPCServer.callback, callback_kwargs)
        if "PYTEST_CURRENT_TEST" not in os.environ:
            # In tests, atexit leads to pytest running forever...
            atexit.register(self._cleanup)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if "add_servicer_callback" in state:
            del state["add_servicer_callback"]

        return state

    def _cleanup(self) -> None:
        if (
            "cleanup_storage_directories_after_shutdown" in self.modyn_config["selector"]
            and self.modyn_config["selector"]["cleanup_storage_directories_after_shutdown"]
        ):
            self.selector_manager.cleanup_trigger_samples()
            self.selector_manager.cleanup_local_storage()

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        super().__exit__(exc_type, exc_val, exc_tb)
        self._cleanup()
        atexit.unregister(self._cleanup)
