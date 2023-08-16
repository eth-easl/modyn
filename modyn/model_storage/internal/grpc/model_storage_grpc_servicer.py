"""Model storage GRPC servicer."""

import logging
import os
import pathlib

import grpc
import torch
from modyn.common.ftp.ftp_utils import download_file, get_pretrained_model_callback
from modyn.model_storage.internal import ModelStorageManager

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    DeleteModelRequest,
    DeleteModelResponse,
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageServicer
from modyn.utils import calculate_checksum, current_time_millis

logger = logging.getLogger(__name__)


class ModelStorageGRPCServicer(ModelStorageServicer):
    """GRPC servicer for the storage module."""

    def __init__(self, config: dict, storage_dir: pathlib.Path, ftp_dir: pathlib.Path):
        """Initialize the model storage GRPC servicer.

        Args:
            config (dict): Configuration of the storage module.
            storage_dir (path): Path to the model storage directory.
            ftp_dir (path): Path to the ftp directory.
        """
        super().__init__()

        self._config = config
        self.ftp_dir = ftp_dir
        self.storage_dir = storage_dir
        self.model_storage_manager = ModelStorageManager(self._config, self.storage_dir)

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
        local_model_path = self.ftp_dir / local_file_name

        logger.info(f"Remote model path is {remote_model_path}, storing at {local_model_path}.")

        success = download_file(
            hostname,
            port,
            "modyn",
            "modyn",
            remote_file_path=pathlib.Path(remote_model_path),
            local_file_path=local_model_path,
            callback=get_pretrained_model_callback(logger),
            checksum=request.checksum,
        )

        if not success:
            logger.error("Downloaded file does not match its checksum.")
            return RegisterModelResponse(success=False)

        logger.info("Download completed. Invoking model storage manager.")

        model_id = self.model_storage_manager.store_model(pipeline_id, trigger_id, local_model_path)
        os.remove(local_model_path)

        return RegisterModelResponse(success=True, model_id=model_id)

    def FetchModel(self, request: FetchModelRequest, context: grpc.ServicerContext) -> FetchModelResponse:
        """Fetch a model from the model storage component.

        Args:
            request: request containing the model id.
            context: the request context.

        Returns:
            FetchModelResponse: the response containing information to download the model.
        """
        logger.info(f"Try to fetch model having id {request.model_id}")

        model_dict = self.model_storage_manager.load_model(request.model_id, request.load_metadata)
        if not model_dict:
            logger.error(f"Trained model {request.model_id} could not be fetched.")
            return FetchModelResponse(success=False)
        model_file_path = self.ftp_dir / f"{current_time_millis()}_{request.model_id}.modyn"
        torch.save(model_dict, model_file_path)

        logger.info(f"Trained model {request.model_id} has local path {model_file_path}")
        return FetchModelResponse(
            success=True,
            model_path=str(model_file_path.relative_to(self.ftp_dir)),
            checksum=calculate_checksum(model_file_path),
        )

    def DeleteModel(self, request: DeleteModelRequest, context: grpc.ServicerContext) -> DeleteModelResponse:
        """Delete model from the model storage component.

        Args:
            request: request for deleting the model.
            context: the request context.

        Returns:
            DeleteModelResponse: the response containing information if the model was found in the database.
        """
        model_id = request.model_id
        logger.info(f"Try to delete model having id {model_id}")

        response = DeleteModelResponse()
        success = self.model_storage_manager.delete_model(model_id)

        if success:
            logger.info(f"Deleted model {request.model_id}.")
        else:
            logger.error(f"Deletion of model {request.model_id} was not successful.")
        response.success = success
        return response
