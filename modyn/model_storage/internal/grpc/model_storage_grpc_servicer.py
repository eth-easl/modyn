"""Model storage GRPC servicer."""

import logging
import pathlib

import grpc
from modyn.common.ftp.ftp_utils import download_file
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

EMIT_MESSAGE_PERCENTAGES = [0.25, 0.5, 0.75]


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

    def RegisterModel(self, request: RegisterModelRequest, context: grpc.ServicerContext) -> RegisterModelResponse:
        """Registers a new model at the model storage component by downloading it from a given server.

        Args:
            request: the requested model.
            context: the request context.

        Returns:
            RegisterModelResponse: the response containing an identifier for the stored model.
        """

        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        hostname, port = request.hostname, request.port
        remote_model_path = request.model_path
        logger.info(f"Try to download model from {hostname}:{port}, pipeline {pipeline_id} and trigger {trigger_id}.")

        local_file_name = f"{current_time_millis()}_{pipeline_id}_{trigger_id}.modyn"
        local_model_path = self.storage_dir / local_file_name

        logger.info(f"Remote model path is {remote_model_path}, storing at {local_model_path}.")

        total_downloaded = 0

        def callback(total_size: int, block_size: int) -> None:
            nonlocal total_downloaded
            perc_before = float(total_downloaded) / total_size
            total_downloaded += block_size
            perc_after = float(total_downloaded) / total_size
            for emit_perc in EMIT_MESSAGE_PERCENTAGES:
                if perc_before <= emit_perc < perc_after:
                    logger.info(f"Completed {emit_perc * 100}% of the download.")

        download_file(
            hostname,
            port,
            "modyn",
            "modyn",
            remote_file_path=pathlib.Path(remote_model_path),
            local_file_path=local_model_path,
            callback=callback,
        )

        logger.info("Download completed.")

        response = RegisterModelResponse()

        with MetadataDatabaseConnection(self._config) as database:
            model_id = database.add_trained_model(pipeline_id, trigger_id, local_file_name)
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
        logger.info(f"Try to fetch model having id {request.model_id}")

        response = FetchModelResponse()
        with MetadataDatabaseConnection(self._config) as database:
            model: TrainedModel = database.session.get(TrainedModel, request.model_id)

            response.model_path = model.model_path
            response.success = True

            logger.info(f"Trained model {request.model_id} has local path {self.storage_dir / model.model_path}")

        return response
