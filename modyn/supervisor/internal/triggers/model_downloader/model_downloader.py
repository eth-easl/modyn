import io
import json
import logging
import pathlib
from typing import Optional

import grpc
import torch
from modyn.common.ftp import download_trained_model
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.models.coreset_methods_support import CoresetSupportingModule
from modyn.utils import dynamic_module_import
from modyn.utils.utils import grpc_connection_established

logger = logging.getLogger(__name__)


# TODO(366) Unify similar code in trainer_server and evaluator. Create common.utils.ModelDownloader
class ModelDownloader:
    def __init__(
        self,
        modyn_config: dict,
        pipeline_id: int,
        device: str,
        base_dir: pathlib.Path,
        model_storage_address: str,
    ):
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self.base_dir = base_dir
        assert self.base_dir.exists(), f"Temporary Directory {self.base_dir} should have been created."

        self._model_storage_stub = self.connect_to_model_storage(model_storage_address)

        self.model_configuration_dict = None
        self.model_handler = None
        self._amp: Optional[bool] = None
        self._model = None

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)

    def _load_state(self, path: pathlib.Path) -> None:
        assert path.exists(), "Cannot load state from non-existing file"

        logger.info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])

        # delete trained model from disk
        path.unlink()

    def configure(self, model_id: int) -> None:
        with MetadataDatabaseConnection(self.modyn_config) as database:
            trained_model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)
            if not trained_model:
                logger.error(f"Trained model {model_id} does not exist!")
                return
            model_class_name, model_config, amp = database.get_model_configuration(trained_model.pipeline_id)
        self._amp = amp
        model_module = dynamic_module_import("modyn.models")
        self.model_handler = getattr(model_module, model_class_name)
        self.model_configuration_dict = json.loads(model_config)
        self._model = self.model_handler(self.model_configuration_dict, self._device, amp)
        assert isinstance(self._model.model, CoresetSupportingModule)

    def download(self, model_id: int) -> None:
        self.configure(model_id)

        fetch_request = FetchModelRequest(model_id=model_id, load_metadata=False)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(f"Trained model {model_id} cannot be fetched from model storage. ")
            raise Exception("Failed to fetch trained model")  # pylint: disable=broad-exception-raised
        trained_model_path = download_trained_model(
            logger=logger,
            model_storage_config=self.modyn_config["model_storage"],
            remote_path=pathlib.Path(fetch_resp.model_path),
            checksum=fetch_resp.checksum,
            identifier=self.pipeline_id,
            base_directory=self.base_dir,
        )
        self._load_state(trained_model_path)
