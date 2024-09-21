from abc import ABC, abstractmethod
from typing import Any

import torch

FULL_GRAD_APPROXIMATION = ["LastLayer", "LastLayerWithEmbedding"]


def get_tensors_subset(
    selected_indexes: list[int], data: torch.Tensor | dict, target: torch.Tensor, sample_ids: list
) -> tuple[torch.Tensor | dict, torch.Tensor]:
    """This function is used in Batch-then-sample.

    The downsampler returns the selected sample ids. We have to work out
    which index the various sample_ids correspond to and then extract
    the selected samples from the tensors. For example, from the
    downsampling strategy we get that the selected ids are 132 and 154
    and that all the ids are [102, 132, 15, 154, 188]. As a result, we
    get that the corresponding ids are 1 and 3 (using in_batch_index),
    and then we get the entries of data and target only for the selected
    samples
    """
    sample_id2index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    # first of all we compute the position of each selected index within the batch
    in_batch_index = [sample_id2index[selected_index] for selected_index in selected_indexes]

    # then we extract the data
    if isinstance(data, torch.Tensor):
        sub_data = data[in_batch_index]
    else:
        sub_data = {key: tensor[in_batch_index] for key, tensor in data.items()}

    # and the targets
    sub_target = target[in_batch_index]

    return sub_data, sub_target


def unsqueeze_dimensions_if_necessary(
    forward_output: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For binary classification, the forward output is a 1D tensor of length
    batch_size. We need to unsqueeze it to have a 2D tensor of shape
    (batch_size, 1).

    For binary classification we use BCEWithLogitsLoss, which requires
    the same dimensionality between the forward output and the target,
    so we also need to unsqueeze the target tensor.
    """
    if forward_output.dim() == 1:
        forward_output = forward_output.unsqueeze(1)
        target = target.unsqueeze(1)
    return forward_output, target


class AbstractRemoteDownsamplingStrategy(ABC):
    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        device: str,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.trigger_id = trigger_id
        self.device = device
        self.modyn_config = modyn_config

        assert "downsampling_ratio" in params_from_selector
        self.downsampling_ratio = params_from_selector["downsampling_ratio"]
        self.ratio_max = params_from_selector["ratio_max"]

        # The next variable is used to keep a mapping index <-> sample_id
        # This is needed since the data selection policy works on indexes (the policy does not care what the sample_id
        # is, it simply stores its score in a vector/matrix) but for retrieving again the data we need somehow to
        # remember the sample_id. So, index_sampleid_map might contain something like [124, 156, 562, 18] and the
        # per-sample score (whatever it is, Gradnom/loss/CRAIG..) be [1.23, 0.31, 14.3, 0.09]. So, for example, the
        # policy selects the two points with highest score ([0, 2]) and we need to know that 0 is sample 124 and 2 is
        # sample 562.
        self.index_sampleid_map: list[int] = []

        # For some strategies, data needs to be supplied class by class in order to get the desired result. If so, you
        # can use the following parameter
        self.requires_data_label_by_label = False

        # Some methods require extra features (embedding recorder, get_last_layer) that are implemented in the class
        # CoresetSupportingModule for model implementations.
        self.requires_coreset_supporting_module = False

        # Some methods might not need information from forward pass (e.g. completely random)
        # Most do (definition), hence we default to True
        # We might want to refactor those downsamplers to presamplers and support some
        # adaptivity at the selector, but for now we allow random downsamplers mostly
        # to support RS2.
        self.forward_required = True

        # Some methods might only support StB, not BtS.
        self.supports_bts = True

    @abstractmethod
    def init_downsampler(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_points(self) -> tuple[list[int], torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def _compute_last_layer_gradient_wrt_loss_sum(
        per_sample_loss_fct: Any, forward_output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient of the last layer with respect to the sum of
        the loss.

        Note: if the gradient with respect to the mean of the loss is needed, the result of this function should be
        divided by the number of samples in the batch.
        """
        if isinstance(per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            # no need to autograd if cross entropy loss is used since closed form solution exists.
            # Because CrossEntropyLoss includes the softmax, we need to apply the
            # softmax to the forward output to obtain the probabilities
            probs = torch.nn.functional.softmax(forward_output, dim=1)
            num_classes = forward_output.shape[-1]

            # Pylint complains torch.nn.functional.one_hot is not callable for whatever reason
            one_hot_targets = torch.nn.functional.one_hot(  # pylint: disable=not-callable
                target, num_classes=num_classes
            )
            last_layer_gradients = probs - one_hot_targets
        else:
            sample_losses = per_sample_loss_fct(forward_output, target)
            last_layer_gradients = torch.autograd.grad(sample_losses.sum(), forward_output, retain_graph=False)[0]
        return last_layer_gradients

    @staticmethod
    def _compute_last_two_layers_gradient_wrt_loss_sum(
        per_sample_loss_fct: Any, forward_output: torch.Tensor, target: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute the gradient of the last two layers with respect to the sum
        of the loss.

        Note: if the gradient with respect to the mean of the loss is needed, the result of this function should be
        divided by the number of samples in the batch.
        """
        loss = per_sample_loss_fct(forward_output, target).sum()
        embedding_dim = embedding.shape[1]
        num_classes = forward_output.shape[1]
        batch_num = target.shape[0]

        with torch.no_grad():
            bias_parameters_grads = torch.autograd.grad(loss, forward_output)[0]
            weight_parameters_grads = embedding.view(batch_num, 1, embedding_dim).repeat(
                1, num_classes, 1
            ) * bias_parameters_grads.view(batch_num, num_classes, 1).repeat(1, 1, embedding_dim)
            gradients = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)
        return gradients
