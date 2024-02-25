from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.models.coreset_methods_support import CoresetSupportingModule
from modyn.metadata_database.models import TrainedModel
from modyn.utils import dynamic_module_import
from modyn.common.ftp import download_trained_model
from modyn.supervisor.internal.triggers.model_wrappers import AbstractModelWrapper

from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub

import grpc
from modyn.utils.utils import (
    grpc_connection_established,
)


import pandas as pd
from typing import Optional, Union
import json
import pathlib
import torch
import io

import logging

logger = logging.getLogger(__name__)


class ModynModelWrapper(AbstractModelWrapper):
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

        # self._info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])

        # delete trained model from disk
        path.unlink()
    
    def configure(self, model_id: int):
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
        self._model = self.model_handler(
            self.model_configuration_dict, self._device, amp
        )
        assert isinstance(self._model.model, CoresetSupportingModule)
    
    def download(self, model_id: int):
        self.configure(model_id)

        fetch_request = FetchModelRequest(model_id=model_id, load_metadata=False)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(
                f"Trained model {model_id} cannot be fetched from model storage. "
            )
            raise Exception
        trained_model_path = download_trained_model(
            logger=logger,
            model_storage_config=self.modyn_config["model_storage"],
            remote_path=pathlib.Path(fetch_resp.model_path),
            checksum=fetch_resp.checksum,
            identifier=self.pipeline_id,
            base_directory=self.base_dir,
        )
        self._load_state(trained_model_path)
    
    def get_embeddings(self, dataloader) -> torch.Tensor:
        assert self._model is not None
        all_embeddings: Optional[torch.Tensor] = None

        self._model.model.eval()
        self._model.model.embedding_recorder.start_recording()

        with torch.no_grad():
            for batch in dataloader:
                data: Union[torch.Tensor, dict]
                if isinstance(batch[1], torch.Tensor):
                    data = batch[1].to(self._device)
                elif isinstance(batch[1], dict):
                    data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                    for name, tensor in batch[1].items():
                        data[name] = tensor.to(self._device)
                else:
                    raise ValueError(f"data type {type(batch[1])} not supported")

                # batch_size = target.shape[0]
                with torch.autocast(self._device_type, enabled=self._amp):
                    output = self._model.model(data)
                    embeddings = self._model.model.embedding_recorder.embedding
                    if all_embeddings is None:
                        all_embeddings = embeddings
                    else:
                        all_embeddings = torch.cat((all_embeddings, embeddings), 0)

        self._model.model.embedding_recorder.end_recording()

        return all_embeddings

    def get_embeddings_evidently_format(self, dataloader) -> pd.DataFrame:
        embeddings_numpy = self.get_embeddings(dataloader).cpu().detach().numpy()
        embeddings_df = pd.DataFrame(embeddings_numpy).astype("float64")
        embeddings_df.columns = ['col_' + str(x) for x in embeddings_df.columns]
        logger.debug(f"[EMBEDDINGS SHAPE] {embeddings_df.shape}")
        # logger.debug(embeddings_df[:3])
        return embeddings_df

