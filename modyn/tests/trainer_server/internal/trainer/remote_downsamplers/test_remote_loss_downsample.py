import torch
from torch import nn

from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    get_tensors_subset,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_loss_downsampling import RemoteLossDownsampling


def test_sample_shape(dummy_system_config: ModynConfig):
    model = torch.nn.Linear(10, 2)
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "ratio_max": 100}
    sampler = RemoteLossDownsampling(
        0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        data = torch.randn(8, 10)
        target = torch.randint(2, size=(8,))
        ids = list(range(8))

        forward_output = model(data)
        sampler.inform_samples(ids, data, forward_output, target)
        indexes, weights = sampler.select_points()

        sampled_data, sampled_target = get_tensors_subset(indexes, data, target, ids)

        assert sampled_data.shape[0] == 4  # (50% of 8)
        assert sampled_data.shape[1] == data.shape[1]
        assert weights.shape[0] == 4
        assert sampled_target.shape[0] == 4
        assert len(indexes) == 4


def test_sample_shape_binary(dummy_system_config: ModynConfig):
    model = torch.nn.Linear(10, 1)
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "ratio_max": 100}
    sampler = RemoteLossDownsampling(
        0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        data = torch.randn(8, 10)
        forward_outputs = model(data).squeeze(1)
        target = torch.randint(2, size=(8,), dtype=torch.float32)
        ids = list(range(8))

        sampler.inform_samples(ids, data, forward_outputs, target)
        downsampled_indexes, weights = sampler.select_points()

        assert len(downsampled_indexes) == 4
        assert weights.shape[0] == 4

        sampled_data, sampled_target = get_tensors_subset(downsampled_indexes, data, target, ids)

        assert weights.shape[0] == sampled_target.shape[0]
        assert sampled_data.shape[0] == 4
        assert sampled_data.shape[1] == data.shape[1]
        assert weights.shape[0] == 4
        assert sampled_target.shape[0] == 4


def test_sample_weights(dummy_system_config: ModynConfig):
    model = torch.nn.Linear(10, 2)
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "ratio_max": 100}
    sampler = RemoteLossDownsampling(
        0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        data = torch.randn(8, 10)
        target = torch.randint(2, size=(8,))
        ids = list(range(8))

        forward_output = model(data)
        sampler.inform_samples(ids, data, forward_output, target)
        selected_ids, weights = sampler.select_points()

        assert weights.sum() > 0
        assert set(selected_ids) <= set(list(range(8)))


# Create a model that always predicts the same class
class AlwaysZeroModel(torch.nn.Module):
    def forward(self, data):
        return torch.zeros(data.shape[0])


def test_sample_loss_dependent_sampling(dummy_system_config: ModynConfig):
    model = AlwaysZeroModel()
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.MSELoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "ratio_max": 100}
    sampler = RemoteLossDownsampling(
        0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        # Create a target with two classes, where half have a true label of 0 and half have a true label of 1
        target = torch.cat([torch.zeros(4), torch.ones(4)])

        # Create a data tensor with four points that have a loss of zero and four points that have a non-zero loss
        data = torch.cat([torch.randn(4, 10), torch.randn(4, 10)], dim=0)

        ids = list(range(8))

        forward_output = model(data)
        sampler.inform_samples(ids, data, forward_output, target)
        indexes, _ = sampler.select_points()

        _, sampled_target = get_tensors_subset(indexes, data, target, ids)

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


def test_sample_dict_input(dummy_system_config: ModynConfig):
    data = {
        "tensor1": torch.randn(6, 10),
        "tensor2": torch.randn(6, 10),
    }
    target = torch.randint(0, 2, size=(6, 8)).float()
    sample_ids = list(range(6))

    # call the sample method with dictionary input
    mymodel = DictLikeModel()
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": 50, "sample_then_batch": False, "ratio_max": 100}
    sampler = RemoteLossDownsampling(
        0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        forward_output = mymodel(data)
        sampler.inform_samples(sample_ids, data, forward_output, target)
        indexes, weights = sampler.select_points()
        sampled_data, sampled_target = get_tensors_subset(indexes, data, target, sample_ids)

        # check that the output has the correct shape and type
        assert isinstance(sampled_data, dict)

        assert all(sampled_data[key].shape == (3, 10) for key in sampled_data)

        assert weights.shape == (3,)
        assert sampled_target.shape == (3, 8)
        assert len(indexes) == 3
        assert set(indexes) <= set(sample_ids)
