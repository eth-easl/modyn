import numpy as np
import torch
from modyn.tests.trainer_server.internal.trainer.remote_downsamplers.deepcore_comparison_tests_utils import DummyModel
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_submodular_downsampling_strategy import (
    RemoteSubmodularDownsamplingStrategy,
)
from torch.nn import BCEWithLogitsLoss


def get_sampler_config(submodular: str = "GraphCut", balance=False):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "submodular_function": submodular,
        "balance": balance,
        "selection_batch": 64,
    }
    return 0, 0, 0, params_from_selector, per_sample_loss_fct, "cpu"


def test_select_different_submodulars():
    _test_select_subm("FacilityLocation")
    _test_select_subm("GraphCut")
    _test_select_subm("LogDeterminant")


def test_select_different_submodulars_balanced():
    _test_select_subm_balance("FacilityLocation")
    _test_select_subm_balance("GraphCut")
    _test_select_subm_balance("LogDeterminant")


def _test_select_subm(submodular, balance=False):
    sampler = RemoteSubmodularDownsamplingStrategy(*get_sampler_config(submodular, balance))
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    assert len(sampler.matrix_elements) == 1
    # 3 samples of dim 5 * 10 + 5
    assert sampler.matrix_elements[0].shape == (3, 55)
    sample_ids = [10, 11, 12, 13]
    forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 1])  # 4 target labels
    embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    assert len(sampler.matrix_elements) == 2
    assert sampler.matrix_elements[0].shape == (3, 55)
    assert sampler.matrix_elements[1].shape == (4, 55)
    assert sampler.index_sampleid_map == [1, 2, 3, 10, 11, 12, 13]
    selected_points, selected_weights = sampler.select_points()
    assert len(selected_points) == 3
    assert len(selected_weights) == 3
    assert all(weight > 0 for weight in selected_weights)
    assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


def _test_select_subm_balance(submodular):
    sampler = RemoteSubmodularDownsamplingStrategy(*get_sampler_config(submodular, True))
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    assert len(sampler.matrix_elements) == 1
    # 3 samples of dim 5 * 10 + 5
    assert sampler.matrix_elements[0].shape == (3, 55)

    sampler.inform_end_of_current_label()
    assert len(sampler.already_selected_weights) == 1
    assert len(sampler.already_selected_samples) == 1
    assert len(sampler.index_sampleid_map) == 0
    assert len(sampler.matrix_elements) == 0

    sample_ids = [10, 11, 12, 13]
    forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 1])  # 4 target labels
    embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    assert len(sampler.matrix_elements) == 1
    assert sampler.matrix_elements[0].shape == (4, 55)
    assert sampler.index_sampleid_map == [10, 11, 12, 13]

    sampler.inform_end_of_current_label()
    assert len(sampler.already_selected_weights) == 3
    assert len(sampler.already_selected_samples) == 3
    assert len(sampler.index_sampleid_map) == 0
    assert len(sampler.matrix_elements) == 0

    selected_points, selected_weights = sampler.select_points()
    assert len(selected_points) == 3
    assert len(selected_weights) == 3
    assert all(weight > 0 for weight in selected_weights)
    assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


def _get_selected_samples(submodular, num_of_target_samples, sample_ids, forward_output, target, embedding):
    np.random.seed(42)

    sampler = RemoteSubmodularDownsamplingStrategy(
        0,
        0,
        5,
        {
            "downsampling_ratio": 10 * num_of_target_samples,
            "submodular_function": submodular,
            "balance": False,
            "selection_batch": 64,
        },
        BCEWithLogitsLoss(reduction="none"),
        "cpu",
    )
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    assert sampler.index_sampleid_map == list(range(10))
    selected_samples, selected_weights = sampler.select_points()
    assert len(selected_samples) == num_of_target_samples
    assert len(selected_weights) == num_of_target_samples
    return selected_samples


def test_matching_with_deepcore():
    torch.manual_seed(23)
    dummy_model = DummyModel()
    samples = torch.rand(10, 1)
    target = torch.tensor([0, 1, 0, 0, 0, 1, 1, 0, 1, 1]).unsqueeze(1).float()
    sample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples).float()
    embedding = dummy_model.embedding

    # Facility Location
    expected = [
        None,
        [6],
        [0, 6],
        [0, 6, 9],
        [0, 2, 6, 9],
        [0, 2, 4, 6, 9],
        [0, 2, 4, 6, 8, 9],
        [0, 1, 2, 4, 6, 8, 9],
        [0, 1, 2, 4, 5, 6, 8, 9],
        [0, 1, 2, 4, 5, 6, 7, 8, 9],
    ]
    for i in range(1, 10):
        assert (
            sorted(_get_selected_samples("FacilityLocation", i, sample_ids, forward_output, target, embedding))
            == expected[i]
        )

    # GraphCut
    expected = [
        None,
        [6],
        [0, 6],
        [0, 6, 8],
        [0, 6, 7, 8],
        [0, 5, 6, 7, 8],
        [0, 4, 5, 6, 7, 8],
        [0, 1, 4, 5, 6, 7, 8],
        [0, 1, 3, 4, 5, 6, 7, 8],
        [0, 1, 3, 4, 5, 6, 7, 8, 9],
    ]
    for i in range(1, 10):
        assert (
            sorted(_get_selected_samples("GraphCut", i, sample_ids, forward_output, target, embedding)) == expected[i]
        )
