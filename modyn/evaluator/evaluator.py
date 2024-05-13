import pathlib
import shutil
import tempfile

from modyn.evaluator.internal.grpc.evaluator_grpc_server import EvaluatorGRPCServer


class Evaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.working_directory = pathlib.Path(tempfile.gettempdir()) / "modyn_evaluator"

        if self.working_directory.exists() and self.working_directory.is_dir():
            shutil.rmtree(self.working_directory)

        self.working_directory.mkdir()

    def run(self) -> None:
        with EvaluatorGRPCServer(self.config, self.working_directory) as server:
            server.wait_for_termination()
        shutil.rmtree(self.working_directory)
