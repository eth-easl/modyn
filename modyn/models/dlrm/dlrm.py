from typing import Any

import numpy as np
import torch
from modyn.models.dlrm.nn.factories import create_interaction
from modyn.models.dlrm.nn.parts import DlrmBottom, DlrmTop
from modyn.models.dlrm.utils.install_lib import install_cuda_extensions_if_not_present
from modyn.models.dlrm.utils.utils import get_device_mapping
from modyn.utils.utils import package_available_and_can_be_imported
from torch import nn


class DLRM:
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = DlrmModel(model_configuration, device, amp)
        self.model.to(device)


class DlrmModel(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        super().__init__()

        self.validate_config(model_configuration, device)

        if (
            model_configuration["embedding_type"] == "joint_sparse"
            or model_configuration["embedding_type"] == "joint_fused"
            or model_configuration["interaction_op"] == "cuda_dot"
        ):
            install_cuda_extensions_if_not_present()

        self._device = device

        # get the sizes of categorical features
        categorical_features_info = model_configuration["categorical_features_info"]
        categorical_features_info_sizes = list(categorical_features_info.values())

        # limit embedding sizes if needed
        if "max_table_size" in model_configuration:
            categorical_features_info_sizes = [
                min(s, model_configuration["max_table_size"]) for s in categorical_features_info_sizes
            ]

        categorical_feature_sizes = np.asarray(categorical_features_info_sizes)
        # Derive ordering of embeddings based on their cardinality
        # Partition embeddings in the GPUs. As we only train with one GPU, all embeddings will be placed there
        # 'device_mapping' contains the embeddings placed on the GPU
        device_mapping = get_device_mapping(categorical_features_info_sizes, num_gpus=1)

        self._embedding_ordering = torch.tensor(device_mapping["embedding"][0]).to(self._device)

        # Get embedding sizes for each GPU
        categorical_feature_sizes = categorical_feature_sizes[device_mapping["embedding"][0]].tolist()

        self._vectors_per_gpu = device_mapping["vectors_per_gpu"]
        self._embedding_device_mapping = device_mapping["embedding"]
        self._embedding_dim = model_configuration["embedding_dim"]
        self._interaction_op = model_configuration["interaction_op"]
        self._hash_indices = model_configuration["hash_indices"]

        interaction = create_interaction(self._interaction_op, len(categorical_feature_sizes), self._embedding_dim, self._device)

        # ignore device here since it is handled by the trainer
        self.bottom_model = DlrmBottom(
            model_configuration["num_numerical_features"],
            categorical_feature_sizes,
            model_configuration["bottom_mlp_sizes"],
            model_configuration["embedding_type"],
            self._embedding_dim,
            hash_indices=self._hash_indices,
            use_cpp_mlp=model_configuration["use_cpp_mlp"],
            fp16=amp,
            device=self._device,
        )

        self.top_model = DlrmTop(
            model_configuration["top_mlp_sizes"], interaction, self._device, use_cpp_mlp=model_configuration["use_cpp_mlp"]
        )

    def validate_config(self, model_configuration: dict[str, Any], device: str) -> None:
        apex_required = (
            model_configuration["embedding_type"] == "joint_fused"
            or model_configuration["embedding_type"] == "joint_sparse"
            or model_configuration["use_cpp_mlp"]
        )
        cuda_required = apex_required or (model_configuration["interaction_op"] == "cuda_dot")

        if cuda_required:
            if "cuda" not in device:
                raise ValueError("The given DLRM configuration requires training on a GPU")
            if not torch.cuda.is_available():
                raise ValueError("The given DLRM configuration requires PyTorch-CUDA support")

        if apex_required:
            if not package_available_and_can_be_imported("apex"):
                raise ValueError("The given DLRM configuration requires NVIDIA APEX to be installed")

    def extra_repr(self) -> str:
        return f"interaction_op={self._interaction_op}, hash_indices={self._hash_indices}"

    def reorder_categorical_input(self, cat_input: torch.Tensor) -> torch.Tensor:
        """Reorder categorical input based on embedding ordering"""
        dim0 = cat_input.shape[0]
        order = self._embedding_ordering.expand(dim0, -1)
        return torch.gather(cat_input, 1, order)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: a dict containing:
                numerical_input (Tensor): with shape [batch_size, num_numerical_features]
                categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
        """
        numerical_input = data["numerical_input"]
        categorical_input = data["categorical_input"]

        from_bottom, bottom_mlp_output = self.bottom_model(
            numerical_input, self.reorder_categorical_input(categorical_input)
        )
        return self.top_model(from_bottom, bottom_mlp_output).squeeze()
