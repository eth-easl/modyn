import torch

from modyn.models.dlrm.dlrm import DLRM


def get_dlrm_configuration():
    return {
        "embedding_dim": 16,
        "interaction_op": "dot",
        "hash_indices": False,
        "bottom_mlp_sizes": [64, 32, 16],
        "top_mlp_sizes": [64, 64, 32, 16, 1],
        "embedding_type": "multi_table",
        "use_cpp_mlp": False,
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
        },
    }


def get_order_list():
    return [19, 0, 21, 9, 20, 10, 22, 11, 1, 4, 2, 23, 14, 3, 6, 13, 7, 17, 15, 24, 8, 25, 18, 12, 5, 16]


def test_dlrm_init():
    model = DLRM(get_dlrm_configuration(), "cpu", False)

    assert torch.equal(model.model._embedding_ordering, torch.tensor(get_order_list()))
    data = {
        "numerical_input": torch.ones((64, 13), dtype=torch.float32),
        "categorical_input": torch.ones((64, 26), dtype=torch.long),
    }
    model.model(data)


def test_dlrm_reorder_categorical_input():
    model = DLRM(get_dlrm_configuration(), "cpu", False)

    test_data = torch.tensor(list(range(26, 52)), dtype=torch.long).expand(64, -1)
    input_data = torch.tensor([x + 26 for x in get_order_list()], dtype=torch.long).expand(64, -1)
    reordered_test_data = model.model.reorder_categorical_input(test_data)
    assert reordered_test_data.shape == (64, 26)
    assert reordered_test_data.dtype == torch.long
    assert torch.equal(reordered_test_data, input_data)


def test_get_last_layer():
    net = DLRM(get_dlrm_configuration(), "cpu", False)
    last_layer = net.model.get_last_layer()

    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.in_features == 16
    assert last_layer.out_features == 1
    assert last_layer.bias.shape == (1,)
    assert last_layer.weight.shape == (1, 16)


def test_dlrm_no_side_effect():
    model = DLRM(get_dlrm_configuration(), "cpu", False)

    data = {
        "numerical_input": torch.ones((64, 13), dtype=torch.float32),
        "categorical_input": torch.ones((64, 26), dtype=torch.long),
    }
    out_off = model.model(data)
    model.model.embedding_recorder.record_embedding = True
    out_on = model.model(data)

    assert torch.equal(out_on, out_off)


def test_shape_embedding_recorder():
    model = DLRM(get_dlrm_configuration(), "cpu", False)

    data = {
        "numerical_input": torch.ones((64, 13), dtype=torch.float32),
        "categorical_input": torch.ones((64, 26), dtype=torch.long),
    }
    model.model(data)
    assert model.model.embedding is None
    model.model.embedding_recorder.record_embedding = True

    last_layer = model.model.get_last_layer()
    recorded_output = model.model(data)
    recorded_embedding = model.model.embedding

    assert recorded_embedding is not None
    assert recorded_embedding.shape == (64, last_layer.in_features)
    assert torch.equal(torch.squeeze(last_layer(recorded_embedding)), recorded_output)
