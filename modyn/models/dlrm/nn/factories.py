from typing import Sequence

from modyn.models.dlrm.nn.embeddings import (
    JointEmbedding, MultiTableEmbeddings, FusedJointEmbedding, JointSparseEmbedding,
    Embeddings
)
from modyn.models.dlrm.nn.interactions import Interaction, CudaDotInteraction, DotInteraction, CatInteraction
from modyn.models.dlrm.nn.mlps import AbstractMlp, TorchMlp, CppMlp

def create_mlp(input_dim: int, sizes: Sequence[int], use_cpp_mlp: bool) -> AbstractMlp:
    return CppMlp(input_dim, sizes) if use_cpp_mlp else TorchMlp(input_dim, sizes)


def create_embeddings(
        embedding_type: str,
        categorical_feature_sizes: Sequence[int],
        embedding_dim: int,
        device: str = "cuda",
        hash_indices: bool = False,
        fp16: bool = False
) -> Embeddings:
    if embedding_type == "joint":
        return JointEmbedding(categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices)
    elif embedding_type == "joint_fused":
        return FusedJointEmbedding(categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices,
                                   amp_train=fp16)
    elif embedding_type == "joint_sparse":
        return JointSparseEmbedding(categorical_feature_sizes, embedding_dim, device=device, hash_indices=hash_indices)
    elif embedding_type == "multi_table":
        return MultiTableEmbeddings(categorical_feature_sizes, embedding_dim,
                                    hash_indices=hash_indices, device=device)
    else:
        raise NotImplementedError(f"unknown embedding type: {embedding_type}")


def create_interaction(interaction_op: str, embedding_num: int, embedding_dim: int) -> Interaction:
    if interaction_op == "dot":
        return DotInteraction(embedding_num, embedding_dim)
    elif interaction_op == "cuda_dot":
        return CudaDotInteraction(
            DotInteraction(embedding_num, embedding_dim)
        )
    elif interaction_op == "cat":
        return CatInteraction(embedding_num, embedding_dim)
    else:
        raise NotImplementedError(f"unknown interaction op: {interaction_op}")