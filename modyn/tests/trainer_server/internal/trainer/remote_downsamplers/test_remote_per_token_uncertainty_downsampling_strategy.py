import pytest
import torch
import numpy as np

from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_per_token_uncertainty_downsampling import (
    RemoteTokenUncertaintyDownsampling,
)

@ pytest.fixture
def base_params():
    return {
        "downsampling_ratio": 50,
        "ratio_max": 100,
        "score_metric": "Entropy",
        "generative": True,
    }

@ pytest.fixture
def dummy_forward_output():
    # Batch size B=2, sequence length T=4, vocab size V=2
    # Zero logits -> uniform softmax -> identical negative entropies
    return torch.zeros((2, 4, 2), dtype=torch.float32)

@ pytest.fixture
def dummy_target():
    # All positions valid (no -100), shape (B, T)
    return torch.zeros((2, 4), dtype=torch.long)

@ pytest.mark.parametrize(
    "weight_per_sample, per_sample_top_k, per_sample_ratio, expected_mask",
    [
        (False, 1, None, [1, 0, 0, 0, 1, 0, 0, 0]),  # 1 token per sample
        (False, None, 50, [1, 1, 0, 0, 1, 1, 0, 0]),  # 50% tokens per sample -> 2 of 4
        (False, None, None, [1, 1, 1, 1, 0, 0, 0, 0]),  # global 50% of 8 tokens -> 4
        (True, 1, None, [1, 1, 1, 1, 0, 0, 0, 0]),  # 1 sample -> all its tokens
        (True, None, 50, [1, 1, 1, 1, 0, 0, 0, 0]),  # 50% of 2 samples -> 1 sample
        (True, None, None, [1, 1, 1, 1, 0, 0, 0, 0]),  # global 50% of samples -> 1
    ],
)
def test_select_points_combinations(
    base_params,
    dummy_forward_output,
    dummy_target,
    weight_per_sample,
    per_sample_top_k,
    per_sample_ratio,
    expected_mask,
):
    # Prepare params
    params = base_params.copy()
    params["weight_per_sample"] = weight_per_sample
    if per_sample_top_k is not None:
        params["per_sample_top_k"] = per_sample_top_k
    if per_sample_ratio is not None:
        params["per_sample_ratio"] = per_sample_ratio

    # Instantiate downsampler
    sampler = RemoteTokenUncertaintyDownsampling(
        pipeline_id=0,
        trigger_id=0,
        batch_size=2,
        params_from_selector=params,
        modyn_config={},
        per_sample_loss=None,
        device="cpu",
    )
    sampler.init_downsampler()
    sampler.inform_samples(
        sample_ids=[0, 1],
        forward_input=None,
        forward_output=dummy_forward_output,
        target=dummy_target,
    )

    token_ids, weights = sampler.select_points()
    # We only care about the weights mask here
    assert weights.tolist() == expected_mask
    assert int(weights.sum().item()) == sum(expected_mask)


def test_invalid_per_sample_combination(base_params):
    # Both top_k and ratio set should raise
    params = base_params.copy()
    params["per_sample_top_k"] = 1
    params["per_sample_ratio"] = 50
    with pytest.raises(ValueError):
        RemoteTokenUncertaintyDownsampling(
            pipeline_id=0,
            trigger_id=0,
            batch_size=2,
            params_from_selector=params,
            modyn_config={},
            per_sample_loss=None,
            device="cpu",
        )
