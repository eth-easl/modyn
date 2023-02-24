# flake8: noqa

import torch

from .dot_based_interact import dotBasedInteract

try:
    import apex

    from .fused_gather_embedding import buckle_embedding_fused_gather
    from .sparse_embedding import JointSparseEmbedding
except Exception as e:
    pass
