import numpy as np
import torch
from modyn.tests.trainer_server.internal.trainer.remote_downsamplers.deepcore_comparison_tests_utils import DummyModel
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_grad_match_downsampling_strategy import (
    RemoteGradMatchDownsamplingStrategy,
)
from torch.nn import BCEWithLogitsLoss


def get_sampler_config():
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "args": {}}
    return 0, 0, 0, params_from_selector, per_sample_loss_fct


def test_select():
    sampler = RemoteGradMatchDownsamplingStrategy(*get_sampler_config())
    sample_ids = [1, 2, 3]
    forward_output = torch.randn(3, 5)  # 3 samples, 5 output classes
    forward_output.requires_grad = True
    target = torch.tensor([1, 1, 1])
    embedding = torch.randn(3, 10)

    sampler.inform_samples(sample_ids, forward_output, target, embedding)

    assert len(sampler.matrix_elements) == 1
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


def test_matching_results_with_deepcore():
    # RESULTS OBTAINED USING DEEPCORE IN THE SAME SETTING (list[i]= result selecting i samples,
    # None when gradmatch is meaningless, so when 0 samples are selected.
    selected_samples_deepcore = [
        None,
        [7],
        [6, 7],
        [2, 6, 7],
        [2, 3, 6, 7],
        [0, 2, 3, 6, 7],
        [0, 1, 2, 3, 6, 7],
        [0, 1, 2, 3, 6, 7],
        [0, 1, 2, 3, 6, 7],
        [0, 1, 2, 3, 6, 7],
    ]
    selected_weights_deepcore = [
        None,
        [1.0],
        [3.407759504625574e-05, 5.847577631357126e-05],
        [5.103206058265641e-05, 3.4257751394761726e-05, 5.825896005262621e-05],
        [5.083320866106078e-05, 4.821651236852631e-05, 3.4427150239935145e-05, 5.805644832435064e-05],
        [
            1.5012016774562653e-05,
            5.077691821497865e-05,
            4.8160465667024255e-05,
            3.447928975219838e-05,
            5.79994848521892e-05,
        ],
        [
            1.5032787814561743e-05,
            5.841296115249861e-06,
            5.079848415334709e-05,
            4.818197339773178e-05,
            3.4458098525647074e-05,
            5.802121086162515e-05,
        ],
        [
            1.5071565940161236e-05,
            5.801955467177322e-06,
            5.083948781248182e-05,
            4.822281334782019e-05,
            3.441966327955015e-05,
            5.806266199215315e-05,
        ],
        [
            1.4997756807133555e-05,
            5.878812771697994e-06,
            5.076439629192464e-05,
            4.8147816414712e-05,
            3.449815994827077e-05,
            5.798730853712186e-05,
        ],
        [
            1.5097687537490856e-05,
            5.778654667665251e-06,
            5.08721532241907e-05,
            4.825499854632653e-05,
            3.440177897573449e-05,
            5.809665162814781e-05,
        ],
    ]

    torch.manual_seed(23)
    dummy_model = DummyModel()
    samples = torch.rand(10, 1)
    target = torch.tensor([0, 1, 0, 0, 0, 1, 1, 0, 1, 1]).unsqueeze(1).float()
    sample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples).float()
    embedding = dummy_model.embedding

    for num_of_target_samples in range(1, 10):
        np.random.seed(42)

        sampler = RemoteGradMatchDownsamplingStrategy(
            0, 0, 5, {"downsampling_ratio": 10 * num_of_target_samples}, BCEWithLogitsLoss(reduction="none")
        )
        sampler.inform_samples(sample_ids, forward_output, target, embedding)
        assert sampler.index_sampleid_map == list(range(10))
        selected_samples, selected_weights = sampler.select_points()
        assert len(selected_samples) == len(selected_weights)
        assert selected_samples_deepcore[num_of_target_samples] == selected_samples
        assert all(
            np.isclose(expected, computed)
            for expected, computed in zip(selected_weights_deepcore[num_of_target_samples], selected_weights)
        )


def test_matching_results_with_deepcore_permutation_fancy_ids():
    index_mapping = [45, 56, 98, 34, 781, 12, 432, 422, 5, 10]
    selected_indices_deepcore = [2, 3, 4, 9]
    selected_samples_deepcore = [index_mapping[i] for i in selected_indices_deepcore]
    selected_weights_deepcore = [
        0.0004691047070082277,
        0.0004625729052349925,
        0.0005646746722050011,
        0.0005694780265912414,
    ]
    torch.manual_seed(467)
    dummy_model = DummyModel()
    np.random.seed(67)
    samples = torch.rand(10, 1)
    targets = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 0, 0]).float().unsqueeze(1)

    sampler = RemoteGradMatchDownsamplingStrategy(
        0, 0, 5, {"downsampling_ratio": 50}, BCEWithLogitsLoss(reduction="none")
    )

    dummy_model.embedding_recorder.start_recording()
    forward_output = dummy_model(samples).float()
    embedding = dummy_model.embedding

    sampler.inform_samples(index_mapping, forward_output, targets, embedding)

    selected_samples, selected_weights = sampler.select_points()

    assert len(selected_samples) == 4
    assert len(selected_weights) == 4
    assert selected_samples_deepcore == selected_samples
    assert all(
        np.isclose(expected, computed) for expected, computed in zip(selected_weights_deepcore, selected_weights)
    )
