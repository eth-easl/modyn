import os
import pathlib
import shutil

from modyn.common.ftp.ftp_server import FTPServer
from modyn.model_storage.internal.grpc.grpc_server import GRPCServer
from modyn.utils import is_directory_writable


class ModelStorage:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._init_model_storage_directory()

    def _init_model_storage_directory(self) -> None:
        model_storage_directory = pathlib.Path(self.config["model_storage"]["models_directory"])

        if not model_storage_directory.exists():
            raise ValueError(
                f"The model storage directory {model_storage_directory} does not exist. \
                  Please create the directory or mount another, existing directory."
            )

        if not is_directory_writable(model_storage_directory):
            raise ValueError(
                f"The model storage directory {model_storage_directory} is not writable. \
                  Please check the directory permissions and try again.\n"
                + f"Directory info: {os.stat(model_storage_directory)}"
            )

        self.models_directory = model_storage_directory / "models"
        self.models_directory.mkdir(exist_ok=True)
        self.ftp_directory = model_storage_directory / "ftp"

        if self.ftp_directory.exists() and self.ftp_directory.is_dir():
            shutil.rmtree(self.ftp_directory)
        self.ftp_directory.mkdir(exist_ok=False)

    def run(self) -> None:
        with GRPCServer(self.config, self.models_directory, self.ftp_directory) as server:
            with FTPServer(self.config["model_storage"]["ftp_port"], self.ftp_directory):
                server.wait_for_termination()

        shutil.rmtree(self.ftp_directory)
