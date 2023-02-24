import torch
from modyn.models.dlrm.dlrm import DLRM


def test_dlrm_init():
    model_configuration = {
        "embedding_dim": 128,
        "interaction_op": "dot",
        "hash_indices": False,
        "bottom_mlp_sizes": [512, 256, 128],
        "top_mlp_sizes": [1024, 1024, 512, 256, 1],
        "embedding_type": "multi_table",
        "use_cpp_mlp": False,
        "fp16": False,
        "bottom_features_ordered": False,
        "device": "cpu",
    }
    model = DLRM(model_configuration)
    data = {
        "numerical_input": torch.ones((64, 13), dtype=torch.float32),
        "categorical_input": torch.ones((64, 26), dtype=torch.long),
    }
    model.model(data)
