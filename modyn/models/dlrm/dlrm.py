from typing import Any

import numpy as np
import torch
from modyn.models.dlrm.nn.factories import create_interaction
from modyn.models.dlrm.nn.parts import DlrmBottom, DlrmTop
from modyn.models.dlrm.utils.feature_spec import get_device_mapping
from modyn.models.dlrm.utils.install_lib import install_cuda_extensions_if_not_present
from torch import nn


class DLRM:
    def __init__(
        self,
        model_configuration: dict[str, Any],
    ) -> None:
        self.model = DlrmModel(model_configuration)
        self.model.to(model_configuration["device"])


class DlrmModel(nn.Module):
    def __init__(
        self,
        model_configuration: dict[str, Any],
    ) -> None:
        super().__init__()

        if (
            model_configuration["embedding_type"] == "joint_sparse"
            or model_configuration["embedding_type"] == "joint_fused"
            or model_configuration["interaction_op"] == "cuda_dot"
        ):
            install_cuda_extensions_if_not_present()

        categorical_features_info = model_configuration["categorical_features_info"]
        categorical_features_info_sizes = list(categorical_features_info.values())

        if "max_table_size" in model_configuration:
            world_embedding_sizes = [min(s, model_configuration["max_table_size"]) for s in categorical_features_info_sizes]
        else:
            world_embedding_sizes = categorical_features_info_sizes

        world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
        device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=1)

        # Embedding sizes for each GPU
        categorical_feature_sizes = world_categorical_feature_sizes[device_mapping["embedding"][0]].tolist()

        self._vectors_per_gpu = device_mapping["vectors_per_gpu"]
        self._embedding_device_mapping = device_mapping["embedding"]
        self._embedding_dim = model_configuration["embedding_dim"]
        self._interaction_op = model_configuration["interaction_op"]
        self._hash_indices = model_configuration["hash_indices"]

        interaction = create_interaction(
            self._interaction_op, len(world_categorical_feature_sizes), self._embedding_dim
        )

        # ignore device here since it is handled by the trainer
        self.bottom_model = DlrmBottom(
            model_configuration["num_numerical_features"],
            categorical_feature_sizes,
            model_configuration["bottom_mlp_sizes"],
            model_configuration["embedding_type"],
            self._embedding_dim,
            hash_indices=self._hash_indices,
            use_cpp_mlp=model_configuration["use_cpp_mlp"],
            fp16=model_configuration["fp16"],
            device=model_configuration["device"],
        )

        self.top_model = DlrmTop(
            model_configuration["top_mlp_sizes"], interaction, use_cpp_mlp=model_configuration["use_cpp_mlp"]
        )

    def extra_repr(self) -> str:
        return f"interaction_op={self._interaction_op}, hash_indices={self._hash_indices}"

    @classmethod
    def from_dict(cls, obj_dict, **kwargs) -> str:  # type: ignore
        """Create from json str"""
        return cls(**obj_dict, **kwargs)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: a dict containing:
                numerical_input (Tensor): with shape [batch_size, num_numerical_features]
                categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
            batch_sizes_per_gpu (Sequence[int]):
        """
        numerical_input = data["numerical_input"]
        categorical_inputs = data["categorical_input"]

        # bottom mlp output may be not present before all to all communication
        from_bottom, bottom_mlp_output = self.bottom_model(numerical_input, categorical_inputs)
        return self.top_model(from_bottom, bottom_mlp_output).squeeze()
