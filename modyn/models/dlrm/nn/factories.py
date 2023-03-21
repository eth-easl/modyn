# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# flake8: noqa

from typing import Sequence

from modyn.models.dlrm.nn.embeddings import (
    Embeddings,
    FusedJointEmbedding,
    JointEmbedding,
    JointSparseEmbedding,
    MultiTableEmbeddings,
)
from modyn.models.dlrm.nn.interactions import CatInteraction, CudaDotInteraction, DotInteraction, Interaction
from modyn.models.dlrm.nn.mlps import AbstractMlp, CppMlp, TorchMlp


def create_mlp(input_dim: int, sizes: Sequence[int], use_cpp_mlp: bool, device: str) -> AbstractMlp:
    return CppMlp(input_dim, sizes, device) if use_cpp_mlp else TorchMlp(input_dim, sizes)


def create_embeddings(
    embedding_type: str,
    categorical_feature_sizes: Sequence[int],
    embedding_dim: int,
    device: str = "cuda",
    hash_indices: bool = False,
    fp16: bool = False,
) -> Embeddings:
    if embedding_type == "joint":
        return JointEmbedding(categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices)
    elif embedding_type == "joint_fused":
        return FusedJointEmbedding(
            categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices, amp_train=fp16
        )
    elif embedding_type == "joint_sparse":
        return JointSparseEmbedding(categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices)
    elif embedding_type == "multi_table":
        return MultiTableEmbeddings(categorical_feature_sizes, embedding_dim, hash_indices=hash_indices, device=device)
    else:
        raise NotImplementedError(f"unknown embedding type: {embedding_type}")


def create_interaction(interaction_op: str, embedding_num: int, embedding_dim: int, device: str) -> Interaction:
    if interaction_op == "dot":
        return DotInteraction(embedding_num, embedding_dim, device)
    elif interaction_op == "cuda_dot":
        return CudaDotInteraction(DotInteraction(embedding_num, embedding_dim, device), device)
    elif interaction_op == "cat":
        return CatInteraction(embedding_num, embedding_dim)
    else:
        raise NotImplementedError(f"unknown interaction op: {interaction_op}")
