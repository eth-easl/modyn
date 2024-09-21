import numpy as np
import pytest
import torch

from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_uncertainty_downsampling_strategy import (
    RemoteUncertaintyDownsamplingStrategy,
)


@pytest.fixture(params=["LeastConfidence", "Entropy", "Margin"])
def sampler_config(dummy_system_config: ModynConfig, request):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "balance": False,
        "score_metric": request.param,
        "ratio_max": 100,
    }
    return 0, 0, 0, params_from_selector, dummy_system_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"


@pytest.fixture(params=[True, False])
def balance_config(request, sampler_config):
    _, _, _, params_from_selector, dummy_system_config, per_sample_loss_fct, device = sampler_config
    params_from_selector["balance"] = request.param
    return 0, 0, 0, params_from_selector, dummy_system_config, per_sample_loss_fct, device


def test_init(sampler_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    assert not amds.requires_coreset_supporting_module
    assert len(amds.scores) == 0
    assert not amds.index_sampleid_map
    assert amds.requires_data_label_by_label == amds.balance


def test_inform_samples(sampler_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    with torch.inference_mode():
        outputs = torch.randn((10, 5))
        sample_ids = list(range(10))
        amds.inform_samples(sample_ids, None, outputs, None)
        assert len(amds.scores) == 10
        assert amds.index_sampleid_map == sample_ids


test_data = {
    "LeastConfidence": {
        "outputs": torch.tensor([[0.1, 0.1, 0.8], [0.3, 0.4, 0.3], [0.33, 0.34, 0.33]]),
        "expected_scores": np.array([0.8, 0.4, 0.34]),  # confidence just picks the highest probability
    },
    "Entropy": {
        "outputs": torch.tensor([[0.1, 0.9], [0.4, 0.6]]),
        "expected_scores": np.array([-0.325, -0.673]),
    },
    "Margin": {
        "outputs": torch.tensor([[0.6, 0.3, 0.1], [0.33, 0.33, 0.34], [0.8, 0.1, 0.1]]),
        "expected_scores": np.array([0.3, 0.01, 0.7]),  # margin between top two classes
    },
}


def test_compute_score(sampler_config):
    metric = sampler_config[3]["score_metric"]
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    outputs = test_data[metric]["outputs"]
    expected_scores = test_data[metric]["expected_scores"]
    scores = amds._compute_score(outputs, disable_softmax=True)
    assert np.allclose(scores, expected_scores, atol=1e-4)


binary_test_data = {
    "LeastConfidence": {
        "outputs": torch.tensor([[-0.8], [0.5], [0.3]]),
        "expected_scores": np.array([0.8, 0.5, 0.3]),  # confidence just picks the highest probability
    },
    "Entropy": {
        "outputs": torch.tensor([[0.8], [0.5], [0.3]]),
        "expected_scores": np.array([-0.5004, -0.6931, -0.6109]),
    },
    "Margin": {
        "outputs": torch.tensor([[0.8], [0.5], [0.3]]),
        "expected_scores": np.array([0.6, 0.0, 0.4]),  # margin between top two classes
    },
}


def test_compute_score_binary(sampler_config):
    metric = sampler_config[3]["score_metric"]
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    outputs = binary_test_data[metric]["outputs"]
    expected_scores = binary_test_data[metric]["expected_scores"]
    scores = amds._compute_score(outputs, disable_softmax=True)
    assert np.allclose(scores, expected_scores, atol=1e-4)


def test_select_points(balance_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*balance_config)
    with torch.inference_mode():
        outputs = torch.randn((10, 5))
        sample_ids = list(range(10))
        amds.inform_samples(sample_ids, None, outputs, None)
        if amds.balance:
            amds.inform_end_of_current_label()
        selected_ids, weights = amds.select_points()
        assert len(selected_ids) == 5
        assert weights.size(0) == 5
        assert set(selected_ids).issubset(set(sample_ids))


def test_select_indexes_from_scores(sampler_config):
    metric = sampler_config[3]["score_metric"]
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)

    # Use the hardcoded outputs from the test_data for the specific metric
    outputs = test_data[metric]["outputs"]
    amds.scores = amds._compute_score(outputs).tolist()
    amds.index_sampleid_map = list(range(len(outputs)))
    target_size = len(outputs) // 2

    selected_indices, _ = amds._select_indexes_from_scores(target_size)

    # Get the expected selected indices by sorting the expected scores
    expected_scores = test_data[metric]["expected_scores"]
    expected_selected_indices = np.argsort(expected_scores)[:target_size].tolist()

    assert selected_indices == expected_selected_indices, f"Failed for metric {metric}"


def test_select_from_scores_shapes(sampler_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    with torch.inference_mode():
        outputs = torch.randn((10, 5))
        sample_ids = list(range(10))
        amds.inform_samples(sample_ids, None, outputs, None)
        selected_ids, weights = amds._select_from_scores()
        assert len(selected_ids) == 5
        assert weights.size(0) == 5
        assert set(selected_ids).issubset(set(sample_ids))


def test_init_downsampler(sampler_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    amds.init_downsampler()
    assert len(amds.scores) == 0
    assert not amds.index_sampleid_map


def test_requires_grad(sampler_config):
    amds = RemoteUncertaintyDownsamplingStrategy(*sampler_config)
    assert not amds.requires_grad
