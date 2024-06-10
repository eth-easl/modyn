import io
import json
import logging
import pathlib
import tempfile
from typing import Any, Optional, Union

import grpc
import torch
from modyn.common.ftp import download_trained_model
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
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
        self.model = IrreducibleLossProducer._load_il_model(modyn_config, rho_pipeline_id, il_model_id, device)
        self.model.model.eval()
        self.loss_cache: dict[int, torch.Tensor] = {}
        self.device = device
        self.per_sample_loss_fct = per_sample_loss

    def get_irreducible_loss(
        self, sample_ids: list[int], forward_input: Union[dict[str, torch.Tensor], torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        # use sample_ids to index into the precomputed loss values. Return the loss values
        # or use forward_input to compute the loss values
        fullly_in_cache = all(sample_id in self.loss_cache for sample_id in sample_ids)
        if fullly_in_cache:
            # torch.stack creates a new tensor
            return torch.stack([self.loss_cache[sample_id] for sample_id in sample_ids]).to(self.device)

        with torch.inference_mode():
            # move model to gpu
            self.model.model.to(self.device)
            forward_output = self.model.model(forward_input)
            irreducible_loss = self.per_sample_loss_fct(forward_output, target).detach()
            cached_loss = irreducible_loss.cpu()
            self.model.model.to("cpu")
        for sample_id, loss in zip(sample_ids, cached_loss):
            self.loss_cache[sample_id] = loss

        return irreducible_loss

    @staticmethod
    def _load_il_model_architecture(modyn_config: dict, rho_pipeline_id: int) -> Any:
        with MetadataDatabaseConnection(modyn_config) as database:
            model_class_name, model_config, amp = database.get_model_configuration(rho_pipeline_id)
        model_config_dict = json.loads(model_config)
        model_module = dynamic_module_import("modyn.models")
        model_handler = getattr(model_module, model_class_name)
        # il model is only moved to gpu when we use it to compute il loss
        model = model_handler(model_config_dict, "cpu", amp)
        return model

    @staticmethod
    def _load_il_model(modyn_config: dict, rho_pipeline_id: int, il_model_id: int, device: str) -> Any:
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
                checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device(device))
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


class RemoteRHOLossDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: torch.nn.modules.loss,
        device: str,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.replacement = False
        self.per_sample_loss_fct = per_sample_loss
        self.irreducible_loss_producer = IrreducibleLossProducer(
            per_sample_loss,
            modyn_config,
            params_from_selector["rho_pipeline_id"],
            params_from_selector["il_model_id"],
            device,
        )

    def init_downsampler(self) -> None:
        self.index_sampleid_map: list[int] = []
        self.rho_loss: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: Union[dict[str, torch.Tensor], torch.Tensor],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        training_loss = self.per_sample_loss_fct(forward_output, target).detach()
        self.index_sampleid_map += sample_ids
        irreducible_loss = self.irreducible_loss_producer.get_irreducible_loss(sample_ids, forward_output, target)

        self.rho_loss.append(training_loss - irreducible_loss)
        self.number_of_points_seen += forward_output.shape[0]

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / 100), 1)
        # find the indices of maximal "target_size" elements in the list of rho_loss
        selected_indices = torch.topk(torch.stack(self.rho_loss), target_size).indices
        selected_sample_ids = [self.index_sampleid_map[i] for i in selected_indices]
        # all selected samples have weight 1.0
        selected_weights = torch.ones(target_size)
        return selected_sample_ids, selected_weights
