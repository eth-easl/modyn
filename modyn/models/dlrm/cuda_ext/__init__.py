# flake8: noqa

import torch

from modyn.utils import package_available_and_can_be_imported

from .dot_based_interact import dotBasedInteract
if package_available_and_can_be_imported("apex"):
    from .fused_gather_embedding import buckle_embedding_fused_gather
    from .sparse_embedding import JointSparseEmbedding