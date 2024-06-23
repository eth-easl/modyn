# pylint: disable=too-many-locals

import torch
from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_grad_match_downsampling_strategy import (
    RemoteGradMatchDownsamplingStrategy,
)


def get_sampler_config(modyn_config: ModynConfig, balance=False):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "balance": balance,
    }
    return 0, 0, 0, params_from_selector, modyn_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"


def test_select(dummy_system_config: ModynConfig):
    sampler = RemoteGradMatchDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [1, 2, 3]
        forward_input = torch.randn(3, 5)  # 3 samples, 5 input features
        forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1])
        embedding = torch.randn(3, 10)

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert len(sampler.matrix_elements) == 1
        assert sampler.matrix_elements[0].shape == forward_output.shape

        sample_ids = [10, 11, 12, 13]
        forward_input = torch.randn(4, 5)  # 4 samples, 5 input features
        forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1, 1])  # 4 target labels
        embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert len(sampler.matrix_elements) == 2
        assert sampler.matrix_elements[0].shape == (3, 5)
        assert sampler.matrix_elements[1].shape == forward_output.shape
        assert sampler.index_sampleid_map == [1, 2, 3, 10, 11, 12, 13]

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 3
        assert len(selected_weights) == 3
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


def test_select_balanced(dummy_system_config: ModynConfig):
    sampler = RemoteGradMatchDownsamplingStrategy(*get_sampler_config(dummy_system_config, True))
    with torch.inference_mode(mode=(not sampler.requires_grad)):

        sample_ids = [1, 2, 3]
        forward_input = torch.randn(3, 5)  # 3 samples, 5 input features
        forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1])
        embedding = torch.randn(3, 10)

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert len(sampler.matrix_elements) == 1
        assert sampler.matrix_elements[0].shape == forward_output.shape

        sampler.inform_end_of_current_label()
        assert len(sampler.matrix_elements) == 0
        assert len(sampler.already_selected_samples) == 1
        assert len(sampler.already_selected_weights) == 1

        sample_ids = [10, 11, 12, 13]
        forward_input = torch.randn(4, 5)  # 4 samples, 5 input features
        forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1, 1])  # 4 target labels
        embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert len(sampler.matrix_elements) == 1
        assert sampler.matrix_elements[0].shape == forward_output.shape
        assert sampler.index_sampleid_map == [10, 11, 12, 13]

        sampler.inform_end_of_current_label()
        assert len(sampler.matrix_elements) == 0
        assert len(sampler.already_selected_samples) == 3
        assert len(sampler.already_selected_weights) == 3

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 3
        assert len(selected_weights) == 3
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)
