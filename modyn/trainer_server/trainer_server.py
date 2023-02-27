import logging
import multiprocessing as mp
import os
import pathlib
import shutil
import tempfile

from modyn.trainer_server.internal.ftp.ftp_server import FTPServer
from modyn.trainer_server.internal.grpc.trainer_server_grpc_server import GRPCServer

logger = logging.getLogger(__name__)


class TrainerServer:
    def __init__(self, config: dict) -> None:
        self.config = config
        try:
            mp.set_start_method("spawn")
        except RuntimeError as error:
            if mp.get_start_method() != "spawn":
                raise error

        self.temp_directory = pathlib.Path(tempfile.gettempdir()) / "modyn"

        if self.temp_directory.exists() and self.temp_directory.is_dir():
            shutil.rmtree(self.temp_directory)

        self.temp_directory.mkdir()

    def run(self) -> None:
        with GRPCServer(self.config, self.temp_directory) as grpc_server:
            with FTPServer(self.config, self.temp_directory):
                grpc_server.wait_for_termination()

        shutil.rmtree(self.temp_directory)
