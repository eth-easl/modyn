import logging
import pathlib
from typing import Optional

import grpc
from modyn.common.ftp import download_trained_model
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.supervisor.internal.triggers.embedding_encoder_utils import EmbeddingEncoder
from modyn.utils.utils import grpc_connection_established

logger = logging.getLogger(__name__)


class EmbeddingEncoderDownloader:
    """The embedding encoder downloader provides a simple interface setup_encoder() to the DataDriftTrigger.
    Given a model_id and a device, it creates an EmbeddingEncoder,
    downloads model parameters and loads model state.
    """

    def __init__(
        self,
        modyn_config: dict,
        pipeline_id: int,
        base_dir: pathlib.Path,
        model_storage_address: str,
    ):
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self.base_dir = base_dir
        assert self.base_dir.exists(), f"Temporary Directory {self.base_dir} should have been created."
        self._model_storage_stub = self.connect_to_model_storage(model_storage_address)

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)

    def configure(self, model_id: int, device: str) -> Optional[EmbeddingEncoder]:
        with MetadataDatabaseConnection(self.modyn_config) as database:
            trained_model: Optional[TrainedModel] = database.session.get(TrainedModel, model_id)
            if not trained_model:
                logger.error(f"Trained model {model_id} does not exist!")
                return None
            model_class_name, model_config, amp = database.get_model_configuration(trained_model.pipeline_id)

        embedding_encoder = EmbeddingEncoder(model_id, model_class_name, model_config, device, amp)
        return embedding_encoder

    def download(self, model_id: int) -> pathlib.Path:
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
        assert trained_model_path is not None
        return trained_model_path

    def setup_encoder(self, model_id: int, device: str) -> EmbeddingEncoder:
        embedding_encoder = self.configure(model_id, device)
        assert embedding_encoder is not None
        trained_model_path = self.download(model_id)
        embedding_encoder._load_state(trained_model_path)
        return embedding_encoder
