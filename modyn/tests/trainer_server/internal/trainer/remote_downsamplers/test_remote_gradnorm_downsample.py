import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    get_tensors_subset,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_gradnorm_downsample import (
    RemoteGradNormDownsampling,
)
from torch import nn


def test_sample_shape_ce():
    model = torch.nn.Linear(10, 3)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    sampler = RemoteGradNormDownsampling(params_from_selector, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(8))
    forward_outputs = model(data)

    downsampled_indexes, weights = sampler.sample(forward_outputs, target)

    assert downsampled_indexes.shape[0] == downsampled_batch_size
    assert weights.shape[0] == downsampled_batch_size

    sampled_data, sampled_target, sampled_ids = get_tensors_subset(downsampled_indexes, data, target, ids)

    assert weights.shape[0] == sampled_target.shape[0]
    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size
    assert set(sampled_ids) <= set(range(8))


def test_sample_shape_other_losses():
    model = torch.nn.Linear(10, 1)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    sampler = RemoteGradNormDownsampling(params_from_selector, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,), dtype=torch.float32).unsqueeze(1)
    ids = list(range(8))

    forward_outputs = model(data)

    downsampled_indexes, weights = sampler.sample(forward_outputs, target)

    assert downsampled_indexes.shape[0] == downsampled_batch_size
    assert weights.shape[0] == downsampled_batch_size

    sampled_data, sampled_target, _ = get_tensors_subset(downsampled_indexes, data, target, ids)

    assert weights.shape[0] == sampled_target.shape[0]
    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size


def test_sampling_crossentropy():
    model = torch.nn.Linear(10, 3)
    downsampled_batch_size = 8
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size, "replacement": False}

    # Here we use autograd since the number of classes is not provided
    sampler = RemoteGradNormDownsampling(params_from_selector, per_sample_loss_fct)
    forward_outputs = model(data)

    _, autograd_weights = sampler.sample(forward_outputs, target)

    # Here we use the closed form shortcut
    sampler = RemoteGradNormDownsampling(params_from_selector, per_sample_loss_fct)

    _, closed_form_weights = sampler.sample(forward_outputs, target)

    # We sort them since in this case sampling is just a permutation (we sample every point without replacement
    autograd_weights, _ = torch.sort(autograd_weights)
    closed_form_weights, _ = torch.sort(closed_form_weights)

    # compare the sorted tensors for equality
    assert torch.all(torch.isclose(closed_form_weights, autograd_weights))


class DictLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 3)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, input_dict):
        first_out = torch.relu(self.fc1(input_dict["tensor1"]))
        second_out = torch.relu(self.fc2(input_dict["tensor2"]))
        output = torch.cat([first_out, second_out], dim=-1)
        return output


def test_sample_dict_input():
    data = {
        "tensor1": torch.randn(6, 10),
        "tensor2": torch.randn(6, 10),
    }
    target = torch.randint(0, 8, size=(6,))
    ids = list(range(10))

    model = DictLikeModel()
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": 3}
    sampler = RemoteGradNormDownsampling(params_from_selector, per_sample_loss_fct)

    forward_outputs = model(data)

    downsampled_indexes, weights = sampler.sample(forward_outputs, target)

    assert downsampled_indexes.shape == (3,)
    assert weights.shape == (3,)

    sampled_data, sampled_target, sampled_ids = get_tensors_subset(downsampled_indexes, data, target, ids)

    # check that the output has the correct shape and type
    assert isinstance(sampled_data, dict)

    assert all(sampled_data[key].shape == (3, 10) for key in sampled_data)

    assert sampled_target.shape == (3,)
    assert len(sampled_ids) == 3
    assert set(sampled_ids) <= set(sampled_ids)
