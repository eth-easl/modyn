import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.rho_loss_utils.irreducible_loss_producer import (
    IrreducibleLossProducer,
)


class RemoteRHOLossDownsampling(AbstractRemoteDownsamplingStrategy):
    """Method adapted from Prioritized Training on Points that are Learnable,
    Worth Learning, and Not Yet Learnt (Sören Mindermann+, 2022).

    https://arxiv.org/abs/2206.07137
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: torch.nn.modules.loss,
        device: str,
        generative: bool = False,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.per_sample_loss_fct = per_sample_loss
        self.irreducible_loss_producer = IrreducibleLossProducer(
            per_sample_loss,
            modyn_config,
            params_from_selector["rho_pipeline_id"],
            params_from_selector["il_model_id"],
            device,
        )
        self.rho_loss: torch.Tensor = torch.tensor([])
        self.number_of_points_seen = 0
        self._device = device

    def init_downsampler(self) -> None:
        self.index_sampleid_map: list[int] = []
        self.rho_loss = torch.tensor([]).to(self._device)
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        training_loss = self.per_sample_loss_fct(forward_output, target).detach()
        self.index_sampleid_map += sample_ids
        irreducible_loss = self.irreducible_loss_producer.get_irreducible_loss(sample_ids, forward_input, target)
        self.rho_loss = torch.cat([self.rho_loss, training_loss - irreducible_loss]).to(training_loss.dtype)
        self.number_of_points_seen += forward_output.shape[0]

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / self.ratio_max), 1)
        # find the indices of maximal "target_size" elements in the list of rho_loss
        selected_indices = torch.topk(self.rho_loss, target_size).indices
        # use sorted() because we keep the relative order of the selected samples
        selected_sample_ids = [self.index_sampleid_map[i] for i in sorted(selected_indices)]
        # all selected samples have weight 1.0
        selected_weights = torch.ones(target_size)
        return selected_sample_ids, selected_weights

    @property
    def requires_grad(self) -> bool:
        return False
