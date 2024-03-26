import logging
from typing import Any

from modyn.common.grpc import GenericGRPCServer
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import (  # noqa: E402, E501
    add_SupervisorServicer_to_server,
)
from modyn.supervisor.internal.grpc.supervisor_grpc_servicer import SupervisorGRPCServicer
from modyn.supervisor.internal.supervisor import Supervisor

logger = logging.getLogger(__name__)


class SupervisorGRPCServer(GenericGRPCServer):
    @staticmethod
    def callback(modyn_config: dict, server: Any, supervisor: Supervisor) -> None:
        add_SupervisorServicer_to_server(SupervisorGRPCServicer(supervisor, modyn_config), server)

    def __init__(self, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self.supervisor = Supervisor(modyn_config)

        callback_kwargs = {"supervisor": self.supervisor}
        super().__init__(
            modyn_config, modyn_config["supervisor"]["port"], SupervisorGRPCServer.callback, callback_kwargs
        )

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Exception) -> None:
        """Exit the context manager.

        Args:
            exc_type (type): exception type
            exc_val (Exception): exception value
            exc_tb (Exception): exception traceback
        """
        super().__exit__(exc_type, exc_val, exc_tb)
        logger.info("SupervisorGRPCServer exited.")
