from typing import Optional
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import \
    AbstractRemoteDownsamplingStrategy


class IrreducibleLossProducer:
    def __init__(self, il_pipeline_id, device: str) -> None:
        # Load the model with the given id
        raise NotImplementedError

    def get_irreducible_loss(self, sample_ids: list[int]) -> torch.Tensor:
        # use sample_ids to index into the precomputed loss values. Return the loss values
        raise NotImplementedError


class RemoteRHOLossDownsampling(AbstractRemoteDownsamplingStrategy):

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        per_sample_loss: torch.nn.modules.loss,
        device: str
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, device)
        self.replacement = False
        self.per_sample_loss_fct = per_sample_loss
        self.irreducible_loss_producer = IrreducibleLossProducer(params_from_selector['il_model_id'], device)


    def init_downsampler(self) -> None:
        self.index_sampleid_map: list[int] = []
        self.rho_loss: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        training_loss = self.per_sample_loss_fct(forward_output, target).detach()
        self.index_sampleid_map += sample_ids
        irreducible_loss = self.irreducible_loss_producer.get_irreducible_loss(sample_ids)
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
