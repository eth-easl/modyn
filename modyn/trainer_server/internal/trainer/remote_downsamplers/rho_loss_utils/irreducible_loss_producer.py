import io
import json
import logging
import pathlib
import tempfile
from typing import Any, Union

import grpc
import torch
from modyn.common.ftp import download_trained_model
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.utils import dynamic_module_import, grpc_connection_established

logger = logging.getLogger(__name__)


class IrreducibleLossProducer:
    def __init__(
        self,
        per_sample_loss: torch.nn.modules.loss,
        modyn_config: dict,
        rho_pipeline_id: int,
        il_model_id: int,
        device: str,
    ) -> None:
        logger.info(f"rho pipeline {rho_pipeline_id} loading irreducible loss model {il_model_id}")
        self.model = IrreducibleLossProducer._load_il_model(modyn_config, rho_pipeline_id, il_model_id)
        self.model.model.eval()
        # We probably will train the main model for multiple epochs.
        # The first time we consume a sample, we compute the irreducible loss from the il model.
        # If we meet this sample again in following epochs, we fetch the loss from cache.
        self.loss_cache: dict[int, torch.Tensor] = {}
        self.device = device
        self.per_sample_loss_fct = per_sample_loss

    def get_irreducible_loss(
        self, sample_ids: list[int], forward_input: Union[dict[str, torch.Tensor], torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        # The time to compute irreducible loss for cache-missing samples should be the same as compute irreducible loss
        # for the full batch. Therefore, to simplify logic as long as there is one cache miss we compute IR loss for the
        # full batch.
        fully_in_cache = all(sample_id in self.loss_cache for sample_id in sample_ids)
        if fully_in_cache:
            # torch.stack creates a new tensor
            return torch.stack([self.loss_cache[sample_id] for sample_id in sample_ids]).to(self.device)

        with torch.inference_mode():
            # move model to the computing device
            self.model.model.to(self.device)
            forward_output = self.model.model(forward_input, sample_ids)
            irreducible_loss = self.per_sample_loss_fct(forward_output, target).detach()
            cached_loss = irreducible_loss.cpu()
            self.model.model.to("cpu")
        for sample_id, loss in zip(sample_ids, cached_loss):
            if sample_id not in self.loss_cache:
                self.loss_cache[sample_id] = loss

        return irreducible_loss

    @staticmethod
    def _load_il_model_architecture(modyn_config: dict, rho_pipeline_id: int) -> Any:
        with MetadataDatabaseConnection(modyn_config) as database:
            model_class_name, model_config, amp = database.get_model_configuration(rho_pipeline_id)
        model_config_dict = json.loads(model_config)
        model_module = dynamic_module_import("modyn.models")
        model_handler = getattr(model_module, model_class_name)
        # il model is only moved to the computing device when we use it to compute il loss
        model = model_handler(model_config_dict, "cpu", amp)
        return model

    @staticmethod
    def _load_il_model(modyn_config: dict, rho_pipeline_id: int, il_model_id: int) -> Any:
        # load the model architecture
        model = IrreducibleLossProducer._load_il_model_architecture(modyn_config, rho_pipeline_id)
        # load the weights
        model_storage_stub = IrreducibleLossProducer.connect_to_model_storage(
            f"{modyn_config['model_storage']['hostname']}:{modyn_config['model_storage']['port']}"
        )
        fetch_request = FetchModelRequest(model_id=il_model_id, load_metadata=True)
        fetch_resp: FetchModelResponse = model_storage_stub.FetchModel(fetch_request)
        assert fetch_resp.success, f"Failed to fetch model with id {il_model_id}"
        with tempfile.TemporaryDirectory() as temp_dir:
            il_model_path = download_trained_model(
                logger=logger,
                model_storage_config=modyn_config["model_storage"],
                remote_path=pathlib.Path(fetch_resp.model_path),
                checksum=fetch_resp.checksum,
                identifier=il_model_id,
                base_directory=pathlib.Path(temp_dir),
            )
            assert il_model_path is not None, f"Failed to download model with id {il_model_id}"
            with open(il_model_path, "rb") as state_file:
                checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))
            assert "model" in checkpoint, f"Model not found in checkpoint for model with id {il_model_id}"
            model.model.load_state_dict(checkpoint["model"])
        return model

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)
