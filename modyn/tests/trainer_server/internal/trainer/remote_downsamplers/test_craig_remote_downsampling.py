import numpy as np
import torch
from modyn.models.coreset_methods_support import CoresetMethodsSupport
from modyn.trainer_server.internal.trainer.remote_downsamplers import RemoteCraigDownsamplingStrategy
from torch import nn
from torch.nn import BCEWithLogitsLoss


def get_sampler_config():
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "args": {}}
    return 0, 0, 0, params_from_selector, per_sample_loss_fct


def test_inform_samples():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    # Test data
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])  # 3 target labels
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    expected_shape = (1, 3, forward_output.shape[1] * (1 + embedding.shape[1]))
    assert len(sampler.current_class_gradients) == 1
    assert np.array(sampler.current_class_gradients).shape == expected_shape


# Dummy distance matrix for testing
initial_matrix = np.array([[0, 1], [1, 0]])


def test_add_to_distance_matrix_single_submatrix():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    submatrix = np.array([[2]])
    sampler.add_to_distance_matrix(initial_matrix)
    sampler.add_to_distance_matrix(submatrix)
    expected_result = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 2]])
    assert np.array_equal(sampler.distance_matrix, expected_result)


def test_add_to_distance_matrix_multiple_submatrix():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    sampler.add_to_distance_matrix(initial_matrix)
    submatrix = np.array([[3, 4], [4, 3]])
    sampler.add_to_distance_matrix(submatrix)
    expected_result = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 3, 4], [0, 0, 4, 3]])
    assert np.array_equal(sampler.distance_matrix, expected_result)


def test_add_to_distance_matrix_large_submatrix():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
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


def test_inform_end_of_current_label_and_select():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)  # 3 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (0, 0)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (3, 3)
    assert len(sampler.current_class_gradients) == 0

    sample_ids = [10, 11, 12, 13]
    forward_output = torch.randn(4, 5)  # 4 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 1])  # 4 target labels
    embedding = torch.randn(4, 10)  # 4 samples, embedding dimension 10

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

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


def test_bts():
    sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    sample_ids = [1, 2, 3, 10, 11, 12, 13]
    forward_output = torch.randn(7, 5)  # 7 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 0, 0, 0, 1])
    embedding = torch.randn(7, 10)  # 7 samples, embedding dimension 10

    assert sampler.distance_matrix.shape == (0, 0)
    sampler.inform_samples(sample_ids, forward_output, target, embedding)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (7, 7)
    assert len(sampler.current_class_gradients) == 0

    assert sampler.index_sampleid_map == [10, 11, 12, 1, 2, 3, 13]

    selected_points, selected_weights = sampler.select_points()

    assert len(selected_points) == 3
    assert len(selected_weights) == 3
    assert all(weight > 0 for weight in selected_weights)
    assert all(id in [1, 2, 3, 10, 11, 12, 13] for id in selected_points)


def test_bts_equals_stb():
    # data
    sample_ids = [1, 2, 3, 10, 11, 12, 13]
    forward_output = torch.randn(7, 5)  # 7 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1, 0, 0, 0, 1])
    embedding = torch.randn(7, 10)  # 7 samples, embedding dimension 10

    # BTS, all in one call
    bts_sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    bts_sampler.inform_samples(sample_ids, forward_output, target, embedding)

    bts_selected_points, bts_selected_weights = bts_sampler.select_points()

    # STB, first class 0 and then class 1
    class0 = target == 0
    class1 = target == 1
    stb_sampler = RemoteCraigDownsamplingStrategy(*get_sampler_config())
    stb_sampler.inform_samples(
        [sample_ids[i] for i, keep in enumerate(class0) if keep],
        forward_output[class0],
        target[class0],
        embedding[class0],
    )
    stb_sampler.inform_end_of_current_label()
    stb_sampler.inform_samples(
        [sample_ids[i] for i, keep in enumerate(class1) if keep],
        forward_output[class1],
        target[class1],
        embedding[class1],
    )
    stb_selected_points, stb_selected_weights = stb_sampler.select_points()

    assert bts_sampler.index_sampleid_map == stb_sampler.index_sampleid_map == [10, 11, 12, 1, 2, 3, 13]
    assert bts_sampler.index_sampleid_map == stb_sampler.index_sampleid_map
    assert stb_selected_points == bts_selected_points
    assert torch.equal(stb_selected_weights, bts_selected_weights)


class DummyModel(CoresetMethodsSupport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer = nn.Linear(in_features=1, out_features=10)
        self.output_layer = nn.Linear(in_features=10, out_features=1)

    def forward(self, input_tensor):
        input_tensor = torch.relu(self.hidden_layer(input_tensor))
        input_tensor = self.embedding_recorder(input_tensor)
        outputs = self.output_layer(input_tensor)
        return outputs

    def get_last_layer(self):
        return self.output_layer


def assert_close_matrices(matrix1, matrix2):
    for row1, row2 in zip(matrix1, matrix2):
        assert len(row1) == len(row2)
        for el1, el2 in zip(row1, row2):
            assert np.isclose(el1, el2, 1e-3)


def test_matching_results_with_deepcore():
    # RESULTS OBTAINED USING DEEPCORE IN THE SAME SETTING
    distance_matrix = [
        [0.17082947127723946, 0.0010000000000000009, 0.06881503558158875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0010000000000000009, 0.17082947127723946, 0.10208309984207153, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.06881503558158875, 0.10208309984207153, 0.17082947127723946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            0.0,
            0.0,
            0.10366127275133385,
            0.04819417542219162,
            0.04819088599085808,
            0.03480425274372101,
            0.06694299152493477,
            0.10332244338840246,
            0.08866848035156727,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.04819417542219162,
            0.10366127275133385,
            0.10366127275133385,
            0.08991418017446995,
            0.013802578508853913,
            0.047890564262866975,
            0.03442031687498093,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.04819088599085808,
            0.10366127275133385,
            0.10360123759508133,
            0.08991729637980461,
            0.013799531221389771,
            0.04788725993037224,
            0.03441711312532425,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.03480425274372101,
            0.08991418017446995,
            0.08991729637980461,
            0.10366127275133385,
            0.0010000000000000009,
            0.03450607305765152,
            0.021281858742237092,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.06694299152493477,
            0.013802578508853913,
            0.013799531221389771,
            0.0010000000000000009,
            0.10366127275133385,
            0.06728020107746124,
            0.08189145086705685,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.10332244338840246,
            0.047890564262866975,
            0.04788725993037224,
            0.03450607305765152,
            0.06728020107746124,
            0.10366127275133385,
            0.0890064619705081,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.08866848035156727,
            0.03442031687498093,
            0.03441711312532425,
            0.021281858742237092,
            0.08189145086705685,
            0.0890064619705081,
            0.10366127275133385,
        ],
    ]
    selected_samples_deepcore = [2, 3]
    selected_weights_deepcore = [4, 8]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    np.random.seed(42)
    samples = torch.tensor(np.random.rand(10, 1).astype(np.float32))
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(0, 0, 5, {"downsampling_ratio": 20}, BCEWithLogitsLoss(reduction="none"))
    sample_ids = [0, 1, 2]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples[0:3]).float()
    target = torch.tensor(targets[0:3]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (0, 0)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (3, 3)
    assert len(sampler.current_class_gradients) == 0

    sample_ids = [3, 4, 5, 6, 7, 8, 9]
    forward_output = dummy_model(samples[3:]).float()
    target = torch.tensor(targets[3:]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (3, 3)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (10, 10)
    assert len(sampler.current_class_gradients) == 0
    assert sampler.index_sampleid_map == list(range(10))

    selected_samples, selected_weights = sampler.select_points()

    assert len(selected_samples) == 2
    assert len(selected_weights) == 2
    assert_close_matrices(distance_matrix, sampler.distance_matrix.tolist())
    assert selected_samples_deepcore == selected_samples
    assert selected_weights_deepcore == selected_weights.tolist()


def test_matching_results_with_deepcore_permutation():
    selected_samples_deepcore = [3, 4, 8]
    selected_weights_deepcore = [3, 2, 8]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    np.random.seed(42)
    samples = torch.tensor(np.random.rand(10, 1).astype(np.float32))
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(0, 0, 5, {"downsampling_ratio": 30}, BCEWithLogitsLoss(reduction="none"))
    sample_ids = [2, 3, 4]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples[targets == 0]).float()
    target = torch.tensor(targets[targets == 0]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (0, 0)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (3, 3)
    assert len(sampler.current_class_gradients) == 0

    sample_ids = [0, 1, 5, 6, 7, 8, 9]
    forward_output = dummy_model(samples[targets == 1]).float()
    target = torch.tensor(targets[targets == 1]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (3, 3)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (10, 10)
    assert len(sampler.current_class_gradients) == 0

    selected_samples, selected_weights = sampler.select_points()

    assert len(selected_samples) == 3
    assert len(selected_weights) == 3
    assert selected_samples_deepcore == selected_samples
    assert selected_weights_deepcore == selected_weights.tolist()


def test_matching_results_with_deepcore_permutation_fancy_ids():
    index_mapping = [45, 56, 98, 34, 781, 12, 432, 422, 5, 10]
    selected_indices_deepcore = [3, 4, 1, 6, 8]
    selected_samples_deepcore = [index_mapping[i] for i in selected_indices_deepcore]
    selected_weights_deepcore = [3, 2, 3, 3, 4]

    torch.manual_seed(42)
    dummy_model = DummyModel()
    np.random.seed(42)
    samples = torch.tensor(np.random.rand(10, 1).astype(np.float32))
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

    sampler = RemoteCraigDownsamplingStrategy(0, 0, 5, {"downsampling_ratio": 50}, BCEWithLogitsLoss(reduction="none"))
    sample_ids = [index_mapping[i] for i in [2, 3, 4]]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples[targets == 0]).float()
    target = torch.tensor(targets[targets == 0]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (0, 0)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (3, 3)
    assert len(sampler.current_class_gradients) == 0

    sample_ids = [index_mapping[i] for i in [0, 1, 5, 6, 7, 8, 9]]
    forward_output = dummy_model(samples[targets == 1]).float()
    target = torch.tensor(targets[targets == 1]).unsqueeze(dim=1).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert sampler.distance_matrix.shape == (3, 3)
    sampler.inform_end_of_current_label()
    assert sampler.distance_matrix.shape == (10, 10)
    assert len(sampler.current_class_gradients) == 0

    selected_samples, selected_weights = sampler.select_points()

    assert len(selected_samples) == 5
    assert len(selected_weights) == 5
    assert selected_samples_deepcore == selected_samples
    assert selected_weights_deepcore == selected_weights.tolist()
