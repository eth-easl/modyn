import os
import pathlib
import shutil
import tempfile

from modyn.common.ftp.ftp_server import FTPServer
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer


class ModelStorage:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._setup_model_storage_directories()

    def _setup_model_storage_directories(self) -> None:
        self.model_storage_directory = pathlib.Path(os.getcwd()) / "model_storage"
        self.ftp_directory = pathlib.Path(tempfile.gettempdir()) / "ftp_model_storage"

        os.makedirs(self.model_storage_directory, exist_ok=True)

        if self.ftp_directory.exists() and self.ftp_directory.is_dir():
            shutil.rmtree(self.ftp_directory)

        self.ftp_directory.mkdir()

    def run(self) -> None:
        with GRPCServer(self.config, self.model_storage_directory, self.ftp_directory) as server:
            with FTPServer(self.config["model_storage"]["ftp_port"], self.ftp_directory):
                server.wait_for_termination()

        shutil.rmtree(self.ftp_directory)
