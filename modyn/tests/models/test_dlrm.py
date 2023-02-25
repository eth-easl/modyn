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
        "device": "cpu",
        "num_numerical_features": 13,
        "categorical_features_info": {
            "cat_0": 7912889,
            "cat_1": 33823,
            "cat_2": 17139,
            "cat_3": 7339,
            "cat_4": 20046,
            "cat_5": 4,
            "cat_6": 7105,
            "cat_7": 1382,
            "cat_8": 63,
            "cat_9": 5554114,
            "cat_10": 582469,
            "cat_11": 245828,
            "cat_12": 11,
            "cat_13": 2209,
            "cat_14": 10667,
            "cat_15": 104,
            "cat_16": 4,
            "cat_17": 968,
            "cat_18": 15,
            "cat_19": 8165896,
            "cat_20": 2675940,
            "cat_21": 7156453,
            "cat_22": 302516,
            "cat_23": 12022,
            "cat_24": 97,
            "cat_25": 35,
        }
    }
    model = DLRM(model_configuration)
    data = {
        "numerical_input": torch.ones((64, 13), dtype=torch.float32),
        "categorical_input": torch.ones((64, 26), dtype=torch.long),
    }
    model.model(data)
