import numpy as np
import torch
from modyn.tests.trainer_server.internal.trainer.remote_downsamplers.deepcore_comparison_tests_utils import DummyModel
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_kcenter_greedy_downsampling_strategy import (
    RemoteKcenterGreedyDownsamplingStrategy,
)
from torch.nn import BCEWithLogitsLoss


def get_sampler_config(balance=False):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "balance": balance,
    }
    return 0, 0, 0, params_from_selector, per_sample_loss_fct


def test_select():
    sampler = RemoteKcenterGreedyDownsamplingStrategy(*get_sampler_config())
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert len(sampler.matrix_elements) == 1
    assert sampler.matrix_elements[0].shape == (3, 10)

    sample_ids = [10, 11, 12, 13]
    forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 1])  # 4 target labels
    embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert len(sampler.matrix_elements) == 2
    assert sampler.matrix_elements[0].shape == (3, 10)
    assert sampler.matrix_elements[1].shape == (4, 10)
    assert sampler.index_sampleid_map == [1, 2, 3, 10, 11, 12, 13]

    selected_points, selected_weights = sampler.select_points()

    assert len(selected_points) == 3
    assert len(selected_weights) == 3
    assert all(weight > 0 for weight in selected_weights)
    assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


def test_select_balanced():
    sampler = RemoteKcenterGreedyDownsamplingStrategy(*get_sampler_config(True))
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert len(sampler.matrix_elements) == 1
    assert sampler.matrix_elements[0].shape == (3, 10)

    sampler.inform_end_of_current_label()
    assert len(sampler.already_selected_samples) == 1
    assert len(sampler.already_selected_weights) == 1

    sample_ids = [10, 11, 12, 13]
    forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 1])  # 4 target labels
    embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert len(sampler.matrix_elements) == 1
    assert sampler.matrix_elements[0].shape == (4, 10)
    assert sampler.index_sampleid_map == [10, 11, 12, 13]
    sampler.inform_end_of_current_label()
    assert len(sampler.already_selected_samples) == 3
    assert len(sampler.already_selected_weights) == 3

    selected_points, selected_weights = sampler.select_points()

    assert len(selected_points) == 3
    assert len(selected_weights) == 3
    assert all(weight > 0 for weight in selected_weights)
    assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)
    assert sum(id in [1, 2, 3] for id in selected_points) == 1
    assert sum(id in [10, 11, 12, 13] for id in selected_points) == 2


def test_matching_results_with_deepcore():
    # RESULTS OBTAINED USING DEEPCORE IN THE SAME SETTING (list[i]= result selecting i samples,
    # None when kcenter is meaningless, so when 0 or 1 samples are selected. 1 is meaningless since kcenter always
    # starts from a random sample)
    selected_samples_deepcore = [
        None,
        None,
        [0, 6],
        [0, 6, 7],
        [0, 3, 6, 7],
        [0, 3, 4, 6, 7],
        [0, 3, 4, 6, 7, 9],
        [0, 2, 3, 4, 6, 7, 9],
        [0, 1, 2, 3, 4, 6, 7, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 9],
    ]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    samples = torch.rand(10, 1)
    target = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1]).float()
    sample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples).float()
    embedding = dummy_model.embedding

    for num_of_target_samples in range(2, 9):
        # reset np seed each time since kcenter starts from a random sample
        np.random.seed(42)

        sampler = RemoteKcenterGreedyDownsamplingStrategy(
            0,
            0,
            5,
            {"downsampling_ratio": 10 * num_of_target_samples, "balance": False},
            BCEWithLogitsLoss(reduction="none"),
        )
        sampler.inform_samples(sample_ids, forward_output, target, embedding)
        assert sampler.index_sampleid_map == list(range(10))
        selected_samples, selected_weights = sampler.select_points()
        assert len(selected_samples) == num_of_target_samples
        assert len(selected_weights) == num_of_target_samples
        assert selected_samples_deepcore[num_of_target_samples] == selected_samples


def test_matching_results_with_deepcore_permutation_fancy_ids():
    index_mapping = [45, 56, 98, 34, 781, 12, 432, 422, 5, 10]
    selected_indices_deepcore = [0, 1, 3, 6, 9]
    selected_samples_deepcore = [index_mapping[i] for i in selected_indices_deepcore]

    torch.manual_seed(467)
    dummy_model = DummyModel()
    np.random.seed(67)
    samples = torch.rand(10, 1)
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 0, 0]).float()

    sampler = RemoteKcenterGreedyDownsamplingStrategy(
        0, 0, 5, {"downsampling_ratio": 50, "balance": False}, BCEWithLogitsLoss(reduction="none")
    )

    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(index_mapping, forward_output, targets, embedding)

    selected_samples, selected_weights = sampler.select_points()

    assert len(selected_samples) == 5
    assert len(selected_weights) == 5
    assert selected_samples_deepcore == selected_samples
