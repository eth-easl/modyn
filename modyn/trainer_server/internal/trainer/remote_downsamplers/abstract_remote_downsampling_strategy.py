from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


def get_tensors_subset(
    selected_indexes: list[int], data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
) -> tuple[Union[torch.Tensor, dict], torch.Tensor]:
    """
    This function is used in Batch-then-sample.
    The downsampler returns the selected sample ids. We have to work out which index the various sample_ids correspond
    to and then extract the selected samples from the tensors.
    For example, from the downsampling strategy we get that the selected ids are 132 and 154 and that all the ids are
    [102, 132, 15, 154, 188]. As a result, we get that the corresponding ids are 1 and 3 (using in_batch_index),
    and then we get the entries of data and target only for the selected samples
    """

    # first of all we compute the position of each selected index within the batch
    in_batch_index = [sample_ids.index(selected_index) for selected_index in selected_indexes]

    # then we extract the data
    if isinstance(data, torch.Tensor):
        sub_data = data[in_batch_index]
    elif isinstance(data, dict):
        sub_data = {key: tensor[in_batch_index] for key, tensor in data.items()}

    # and the targets
    sub_target = target[in_batch_index]

    return sub_data, sub_target


class AbstractRemoteDownsamplingStrategy(ABC):
    def __init__(
            self,
            pipeline_id: int,
            trigger_id: int,
            batch_size: int,
            params_from_selector: dict,
            modyn_config: dict,
            device: str
    ) -> None:
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.trigger_id = trigger_id
        self.device = device
        self.modyn_config = modyn_config

        assert "downsampling_ratio" in params_from_selector
        self.downsampling_ratio = params_from_selector["downsampling_ratio"]

        self.replacement = params_from_selector.get("replacement", True)

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
        forward_input: Union[dict[str, torch.Tensor], torch.Tensor],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_points(self) -> tuple[list[int], torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        raise NotImplementedError
