from modyn.trainer_server.internal.grpc.grpc_server import GRPCServer


class TrainerServer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self) -> None:
        with GRPCServer(self.config) as server:
            server.wait_for_termination()
