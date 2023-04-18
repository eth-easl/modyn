import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    get_tensors_subset,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_loss_downsample import RemoteLossDownsampling
from torch import nn


def test_sample_shape():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    sampler = RemoteLossDownsampling(params_from_selector, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(8))

    forward_output = model(data)
    indexes, weights = sampler.sample(forward_output, target)
    sampled_data, sampled_target, sampled_ids = get_tensors_subset(indexes, data, target, ids)

    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size
    assert len(sampled_ids) == 5


def test_sample_weights():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    sampler = RemoteLossDownsampling(params_from_selector, per_sample_loss_fct)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(8))

    forward_output = model(data)
    _, weights = sampler.sample(forward_output, target)

    assert weights.sum() > 0
    assert set(ids) <= set(list(range(8)))


# Create a model that always predicts the same class
class AlwaysZeroModel(torch.nn.Module):
    def forward(self, data):
        return torch.zeros(data.shape[0])


def test_sample_loss_dependent_sampling():
    model = AlwaysZeroModel()
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.MSELoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    sampler = RemoteLossDownsampling(params_from_selector, per_sample_loss_fct)

    # Create a target with two classes, where half have a true label of 0 and half have a true label of 1
    target = torch.cat([torch.zeros(4), torch.ones(4)])

    # Create a data tensor with four points that have a loss of zero and four points that have a non-zero loss
    data = torch.cat([torch.randn(4, 10), torch.randn(4, 10)], dim=0)

    ids = list(range(8))

    forward_output = model(data)
    indexes, _ = sampler.sample(forward_output, target)
    _, sampled_target, _ = get_tensors_subset(indexes, data, target, ids)

    # Assert that no points with a loss of zero were selected
    assert (sampled_target == 0).sum() == 0
    assert (sampled_target > 0).sum() > 0


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
    target = torch.randint(0, 2, size=(6, 8)).float()
    sample_ids = list(range(10))

    # call the sample method with dictionary input
    mymodel = DictLikeModel()
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": 3}
    sampler = RemoteLossDownsampling(params_from_selector, per_sample_loss_fct)

    forward_output = mymodel(data)
    indexes, weights = sampler.sample(forward_output, target)
    sampled_data, sampled_target, sampled_ids = get_tensors_subset(indexes, data, target, sample_ids)

    # check that the output has the correct shape and type
    assert isinstance(sampled_data, dict)

    assert all(sampled_data[key].shape == (3, 10) for key in sampled_data)

    assert weights.shape == (3,)
    assert sampled_target.shape == (3, 8)
    assert len(sampled_ids) == 3
    assert set(sampled_ids) <= set(sample_ids)
