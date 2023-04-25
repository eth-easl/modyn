"""Model storage GRPC servicer."""

import logging
import pathlib
from ftplib import FTP
from typing import Any

import enlighten
import grpc
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models.trained_models import TrainedModel

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageServicer
from modyn.utils import current_time_millis

logger = logging.getLogger(__name__)


class ModelStorageGRPCServicer(ModelStorageServicer):
    """GRPC servicer for the storage module."""

    def __init__(self, config: dict, storage_dir: pathlib.Path):
        """Initialize the model storage GRPC servicer.

        Args:
            config (dict): Configuration of the storage module.
        """
        super().__init__()

        self._config = config
        self.storage_dir = storage_dir
        self.progress_mgr = enlighten.get_manager()

    def RegisterModel(self, request: RegisterModelRequest, context: grpc.ServicerContext) -> RegisterModelResponse:
        """Registers a new model at the model storage component by downloading it from the trainer server.

        Args:
            request: the requested model.
            context: the request context.

        Returns:
            RegisterModelResponse: the response containing an identifier for the stored model.
        """

        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        logger.info(f"Try to store model from pipeline {pipeline_id} and trigger {trigger_id}")

        remote_model_path = f"/{request.model_path}"
        local_model_path = self.storage_dir / f"{current_time_millis()}.modyn"

        ftp = FTP()
        ftp.connect(
            self._config["trainer_server"]["hostname"], int(self._config["trainer_server"]["ftp_port"]), timeout=3
        )

        ftp.login("modyn", "modyn")
        ftp.sendcmd("TYPE i")  # Switch to binary mode
        size = ftp.size(remote_model_path)

        pbar = self.progress_mgr.counter(total=size, desc=f"[Pipeline {pipeline_id}] Downloading Model", unit="bytes")

        logger.info(
            f"Remote model path is {remote_model_path}, storing at {local_model_path}."
            + f"Fetching via FTP! Total size = {size} bytes."
        )

        with open(local_model_path, "wb") as local_file:

            def write_callback(data: Any) -> None:
                local_file.write(data)
                pbar.update(min(len(data), pbar.total - pbar.count))

            ftp.retrbinary(f"RETR {remote_model_path}", write_callback)

        ftp.close()
        pbar.update(pbar.total - pbar.count)
        pbar.clear(flush=True)
        pbar.close(clear=True)

        logger.info("Wrote model to disk.")

        response = RegisterModelResponse()

        with MetadataDatabaseConnection(self._config) as database:
            model_id = database.add_trained_model(pipeline_id, trigger_id, str(local_model_path))
            response.model_id = model_id
            response.success = True

        return response

    def FetchModel(self, request: FetchModelRequest, context: grpc.ServicerContext) -> FetchModelResponse:
        """Fetch a model from the model storage component.

        Args:
            request: request containing the model id.
            context: the request context.

        Returns:
            FetchModelResponse: the response containing information to download the model.
        """
        logger.info(f"Try to fetch model with id {request.model_id}")

        response = FetchModelResponse()
        with MetadataDatabaseConnection(self._config) as database:
            model: TrainedModel = database.session.get(TrainedModel, request.model_id)

            response.model_path = model.model_path
            response.valid = True

            logger.info(f"Trained model {request.model_id} has local path {model.model_path}")

        return response
