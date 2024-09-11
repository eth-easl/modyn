# pylint: disable=too-many-locals
import numpy as np
import pytest
import torch
from torch.nn import BCEWithLogitsLoss

from modyn.config import ModynConfig
from modyn.tests.trainer_server.internal.trainer.remote_downsamplers.deepcore_comparison_tests_utils import (
    DummyModel,
    assert_close_matrices,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers import RemoteCraigDownsamplingStrategy


def get_sampler_config(modyn_config, balance=False, grad_approx="LastLayerWithEmbedding"):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "balance": balance,
        "selection_batch": 64,
        "greedy": "NaiveGreedy",
        "full_grad_approximation": grad_approx,
        "ratio_max": 100,
    }
    return 0, 0, 0, params_from_selector, modyn_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"


def test_inform_samples(dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        # Test data
        sample_ids = [1, 2, 3]
        forward_input = torch.randn(3, 5)  # 3 samples, 5 input features
        forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1])  # 3 target labels
        embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        expected_shape = (1, 3, forward_output.shape[1] * (1 + embedding.shape[1]))
        assert len(sampler.current_class_gradients) == 1
        assert np.array(sampler.current_class_gradients).shape == expected_shape


# Dummy distance matrix for testing
initial_matrix = np.array([[0, 1], [1, 0]])


def test_add_to_distance_matrix_single_submatrix(dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        submatrix = np.array([[2]])
        sampler.add_to_distance_matrix(initial_matrix)
        sampler.add_to_distance_matrix(submatrix)
        expected_result = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
        assert np.array_equal(sampler.distance_matrix, expected_result)


def test_add_to_distance_matrix_multiple_submatrix(dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sampler.add_to_distance_matrix(initial_matrix)
        submatrix = np.array([[3, 4], [4, 3]])
        sampler.add_to_distance_matrix(submatrix)
        expected_result = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 3, 4], [0, 0, 4, 3]])
        assert np.array_equal(sampler.distance_matrix, expected_result)


def test_add_to_distance_matrix_large_submatrix(dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sampler.add_to_distance_matrix(initial_matrix)
        submatrix = np.array([[5, 6, 7], [6, 5, 7], [7, 7, 5]])
        sampler.add_to_distance_matrix(submatrix)
        sampler.add_to_distance_matrix(np.array([[0, 0], [0, 0]]))
        expected_result = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 5, 6, 7, 0, 0],
                [0, 0, 6, 5, 7, 0, 0],
                [0, 0, 7, 7, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        assert np.array_equal(sampler.distance_matrix, expected_result)


@pytest.mark.parametrize("grad_approx", ["LastLayerWithEmbedding", "LastLayer"])
def test_inform_end_of_current_label_and_select(grad_approx: str, dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config, grad_approx=grad_approx))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [1, 2, 3]
        forward_input = torch.randn(3, 5)  # 3 samples, 5 input features
        forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1])
        embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (3, 3)
        assert len(sampler.current_class_gradients) == 0

        sample_ids = [10, 11, 12, 13]
        forward_input = torch.randn(4, 5)  # 4 samples, 5 input features
        forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([0, 0, 0, 0])  # 4 target labels
        embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (3, 3)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (7, 7)
        assert len(sampler.current_class_gradients) == 0
        assert sampler.index_sampleid_map == [1, 2, 3, 10, 11, 12, 13]

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 3
        assert len(selected_weights) == 3
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


@pytest.mark.parametrize("grad_approx", ["LastLayerWithEmbedding", "LastLayer"])
def test_inform_end_of_current_label_and_select_balanced(grad_approx: str, dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config, True, grad_approx=grad_approx))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [1, 2, 3, 4]
        forward_input = torch.randn(4, 5)
        forward_output = torch.randn(4, 5)
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1, 1])
        embedding = torch.randn(4, 10)

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert len(sampler.already_selected_samples) == 2
        assert len(sampler.already_selected_weights) == 2
        assert sampler.distance_matrix.shape == (0, 0)
        assert len(sampler.current_class_gradients) == 0

        sample_ids = [10, 11, 12, 13, 14, 15]
        forward_output = torch.randn(6, 5)
        forward_output = torch.randn(6, 5)  # 4 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([0, 0, 0, 0, 0, 0])  # 4 target labels
        embedding = torch.randn(6, 10)  # 4 samples, embedding dimension 10

        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert len(sampler.already_selected_samples) == 5
        assert len(sampler.already_selected_weights) == 5
        assert sampler.distance_matrix.shape == (0, 0)
        assert len(sampler.current_class_gradients) == 0

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 5
        assert len(selected_weights) == 5
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 4, 10, 11, 12, 13, 14, 15] for id in selected_points)
        assert sum(id in [1, 2, 3, 4] for id in selected_points) == 2
        assert sum(id in [10, 11, 12, 13, 14, 15] for id in selected_points) == 3


@pytest.mark.parametrize("grad_approx", ["LastLayerWithEmbedding", "LastLayer"])
def test_bts(grad_approx: str, dummy_system_config: ModynConfig):
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config, grad_approx=grad_approx))
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [1, 2, 3, 10, 11, 12, 13]
        forward_input = torch.randn(7, 5)  # 7 samples, 5 input features
        forward_output = torch.randn(7, 5)  # 7 samples, 5 output classes
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1, 0, 0, 0, 1])
        embedding = torch.randn(7, 10)  # 7 samples, embedding dimension 10

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (7, 7)
        assert len(sampler.current_class_gradients) == 0

        assert sampler.index_sampleid_map == [10, 11, 12, 1, 2, 3, 13]

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 3
        assert len(selected_weights) == 3
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


@pytest.mark.parametrize("grad_approx", ["LastLayer", "LastLayerWithEmbedding"])
def test_bts_binary(grad_approx: str, dummy_system_config: ModynConfig):
    sampler_config = get_sampler_config(dummy_system_config, grad_approx=grad_approx)
    per_sample_loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
    sampler_config = (0, 0, 0, sampler_config[3], sampler_config[4], per_sample_loss_fct, "cpu")
    sampler = RemoteCraigDownsamplingStrategy(*sampler_config)

    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [1, 2, 3, 10, 11, 12, 13]
        forward_input = torch.randn(7, 5)  # 7 samples, 5 input features
        forward_output = torch.randn(
            7,
        )
        forward_output.requires_grad = True
        target = torch.tensor([1, 1, 1, 0, 0, 0, 1], dtype=torch.float32)  # 7 target labels
        embedding = torch.randn(7, 10)  # 7 samples, embedding dimension 10

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (7, 7)
        assert len(sampler.current_class_gradients) == 0

        assert sampler.index_sampleid_map == [10, 11, 12, 1, 2, 3, 13]

        selected_points, selected_weights = sampler.select_points()

        assert len(selected_points) == 3
        assert len(selected_weights) == 3
        assert all(weight > 0 for weight in selected_weights)
        assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


@pytest.mark.parametrize("grad_approx", ["LastLayerWithEmbedding", "LastLayer"])
def test_bts_equals_stb(grad_approx: str, dummy_system_config: ModynConfig):
    # data
    sample_ids = [1, 2, 3, 10, 11, 12, 13]
    forward_input = torch.randn(7, 5)  # 7 samples, 5 input features
    forward_output = torch.randn(7, 5)  # 7 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 0, 0, 0, 1])
    embedding = torch.randn(7, 10)  # 7 samples, embedding dimension 10

    # BTS, all in one call
    bts_sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config, grad_approx=grad_approx))
    with torch.inference_mode(mode=(not bts_sampler.requires_grad)):
        bts_sampler.inform_samples(sample_ids, forward_input, forward_output, target, embedding)

        bts_selected_points, bts_selected_weights = bts_sampler.select_points()

        # STB, first class 0 and then class 1
        class0 = target == 0
        class1 = target == 1
        stb_sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config(dummy_system_config, grad_approx=grad_approx))
        stb_sampler.inform_samples(
            [sample_ids[i] for i, keep in enumerate(class0) if keep],
            forward_input[class0],
            forward_output[class0],
            target[class0],
            embedding[class0],
        )
        stb_sampler.inform_end_of_current_label()
        stb_sampler.inform_samples(
            [sample_ids[i] for i, keep in enumerate(class1) if keep],
            forward_input[class1],
            forward_output[class1],
            target[class1],
            embedding[class1],
        )
        stb_selected_points, stb_selected_weights = stb_sampler.select_points()

        assert bts_sampler.index_sampleid_map == stb_sampler.index_sampleid_map == [10, 11, 12, 1, 2, 3, 13]
        assert bts_sampler.index_sampleid_map == stb_sampler.index_sampleid_map
        assert stb_selected_points == bts_selected_points
        assert torch.equal(stb_selected_weights, bts_selected_weights)


def test_matching_results_with_deepcore(dummy_system_config: ModynConfig):
    # RESULTS OBTAINED USING DEEPCORE IN THE SAME SETTING
    expected_distance_matrix = [
        [0.23141611747646584, 0.0010000000000000009, 0.08913177049160004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0010000000000000009, 0.23141611747646584, 0.13688503003120422, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.08913177049160004, 0.13688503003120422, 0.23141611747646584, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.0,
            0.07399794256687164,
            0.013805931270122529,
            0.04436375929415226,
            0.0010000000000000009,
            0.047306565403938294,
            0.07344558201637119,
            0.037069154739379884,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.013805931270122529,
            0.07405797772312417,
            0.042330792009830476,
            0.06103372650593519,
            0.03912345489859581,
            0.014334088027477265,
            0.050075888469815255,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.04436375929415226,
            0.042330792009830476,
            0.07399794256687164,
            0.029320956975221635,
            0.07084722916968167,
            0.044916815891861916,
            0.06630006742104888,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0010000000000000009,
            0.06103372650593519,
            0.029320956975221635,
            0.07397266097884858,
            0.026116726756095887,
            0.0015217790603637704,
            0.03705782613158226,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.047306565403938294,
            0.03912345489859581,
            0.07084722916968167,
            0.026116726756095887,
            0.07405797772312417,
            0.04786216451227665,
            0.0630889374986291,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.07344558201637119,
            0.014334088027477265,
            0.044916815891861916,
            0.0015217790603637704,
            0.04786216451227665,
            0.07399794256687164,
            0.03761516308784485,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.037069154739379884,
            0.050075888469815255,
            0.06630006742104888,
            0.03705782613158226,
            0.0630889374986291,
            0.03761516308784485,
            0.07405797772312417,
        ],
    ]

    selected_samples_deepcore = [2, 5]
    selected_weights_deepcore = [4, 8]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    np.random.seed(42)
    samples = torch.rand(10, 1)
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(
        0,
        0,
        5,
        {
            "downsampling_ratio": 20,
            "balance": False,
            "selection_batch": 64,
            "greedy": "NaiveGreedy",
            "full_grad_approximation": "LastLayerWithEmbedding",
            "ratio_max": 100,
        },
        dummy_system_config.model_dump(by_alias=True),
        BCEWithLogitsLoss(reduction="none"),
        "cpu",
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [0, 1, 2]
        dummy_model.embedding_recorder.start_recording()
        forward_output = dummy_model(samples[0:3]).float()
        target = torch.tensor(targets[0:3]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[0:3], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (3, 3)
        assert len(sampler.current_class_gradients) == 0

        sample_ids = [3, 4, 5, 6, 7, 8, 9]
        forward_output = dummy_model(samples[3:]).float()
        target = torch.tensor(targets[3:]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[3:], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (3, 3)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (10, 10)
        assert len(sampler.current_class_gradients) == 0
        assert sampler.index_sampleid_map == list(range(10))

        selected_samples, selected_weights = sampler.select_points()

        assert len(selected_samples) == 2
        assert len(selected_weights) == 2
        assert_close_matrices(expected_distance_matrix, sampler.distance_matrix.tolist())
        assert selected_samples_deepcore == selected_samples
        assert selected_weights_deepcore == selected_weights.tolist()


def test_matching_results_with_deepcore_permutation(dummy_system_config: ModynConfig):
    selected_samples_deepcore = [2, 1, 5]
    selected_weights_deepcore = [4, 3, 6]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    np.random.seed(42)
    samples = torch.rand(10, 1)
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(
        0,
        0,
        5,
        {
            "downsampling_ratio": 30,
            "balance": False,
            "selection_batch": 64,
            "greedy": "NaiveGreedy",
            "full_grad_approximation": "LastLayerWithEmbedding",
            "ratio_max": 100,
        },
        dummy_system_config.model_dump(by_alias=True),
        BCEWithLogitsLoss(reduction="none"),
        "cpu",
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [2, 3, 4]
        dummy_model.embedding_recorder.start_recording()
        forward_output = dummy_model(samples[targets == 0]).float()
        target = torch.tensor(targets[targets == 0]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[targets == 0], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (3, 3)
        assert len(sampler.current_class_gradients) == 0

        sample_ids = [0, 1, 5, 6, 7, 8, 9]
        forward_output = dummy_model(samples[targets == 1]).float()
        target = torch.tensor(targets[targets == 1]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[targets == 1], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (3, 3)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (10, 10)
        assert len(sampler.current_class_gradients) == 0

        selected_samples, selected_weights = sampler.select_points()

        assert len(selected_samples) == 3
        assert len(selected_weights) == 3
        assert selected_samples_deepcore == selected_samples
        assert selected_weights_deepcore == selected_weights.tolist()


def test_matching_results_with_deepcore_permutation_fancy_ids(dummy_system_config: ModynConfig):
    index_mapping = [45, 56, 98, 34, 781, 12, 432, 422, 5, 10]
    selected_indices_deepcore = [2, 3, 4, 1, 9]
    selected_samples_deepcore = [index_mapping[i] for i in selected_indices_deepcore]
    # This test is a bit flaky - probably due to numerical issues. Sometimes, index 5 is selected instead of 1
    selected_indices_deepcore2 = [2, 3, 4, 5, 9]
    selected_samples_deepcore2 = [index_mapping[i] for i in selected_indices_deepcore2]
    selected_weights_deepcore = [2, 2, 2, 3, 6]

    torch.manual_seed(2)
    dummy_model = DummyModel()
    np.random.seed(3)
    samples = torch.rand(10, 1)
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(
        0,
        0,
        5,
        {
            "downsampling_ratio": 50,
            "balance": False,
            "selection_batch": 64,
            "greedy": "NaiveGreedy",
            "full_grad_approximation": "LastLayerWithEmbedding",
            "ratio_max": 100,
        },
        dummy_system_config.model_dump(by_alias=True),
        BCEWithLogitsLoss(reduction="none"),
        "cpu",
    )
    with torch.inference_mode(mode=(not sampler.requires_grad)):
        sample_ids = [index_mapping[i] for i in [2, 3, 4]]
        dummy_model.embedding_recorder.start_recording()
        forward_output = dummy_model(samples[targets == 0]).float()
        target = torch.tensor(targets[targets == 0]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[targets == 0], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (0, 0)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (3, 3)
        assert len(sampler.current_class_gradients) == 0

        sample_ids = [index_mapping[i] for i in [0, 1, 5, 6, 7, 8, 9]]
        forward_output = dummy_model(samples[targets == 1]).float()
        target = torch.tensor(targets[targets == 1]).unsqueeze(dim=1).float()
        embedding = dummy_model.embedding

        sampler.inform_samples(sample_ids, samples[targets == 1], forward_output, target, embedding)

        assert sampler.distance_matrix.shape == (3, 3)
        sampler.inform_end_of_current_label()
        assert sampler.distance_matrix.shape == (10, 10)
        assert len(sampler.current_class_gradients) == 0

        selected_samples, selected_weights = sampler.select_points()

        assert len(selected_samples) == 5
        assert len(selected_weights) == 5

        # Allow for flakiness with two options
        assert selected_samples in (selected_samples_deepcore, selected_samples_deepcore2)
        assert selected_weights_deepcore == selected_weights.tolist()
