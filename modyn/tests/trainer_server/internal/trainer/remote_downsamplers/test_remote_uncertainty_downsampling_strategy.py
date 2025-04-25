import pytest
import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_token_uncertainty_downsampling import (
    RemoteTokenUncertaintyDownsampling,
)


@pytest.fixture

def base_params():
    return {
        "downsampling_ratio": 50,
        "ratio_max": 100,
        "score_metric": "Entropy",
        "generative": True,
    }


@pytest.fixture

def dummy_uniform_forward_output():
    # B=2, T=4, V=2: uniform logits -> identical entropy
    return torch.zeros((2, 4, 2), dtype=torch.float32)


@pytest.fixture

def dummy_varied_forward_output():
    # B=2, T=4, V=2: varied logits so entropy differs per token
    # Sample 0: tokens [uniform, peaked, uniform, peaked]
    # Sample 1: tokens [peaked, uniform, peaked, uniform]
    forward = torch.zeros((2, 4, 2), dtype=torch.float32)
    forward[0, 1] = torch.tensor([10.0, 0.0])
    forward[0, 3] = torch.tensor([0.0, 10.0])
    forward[1, 0] = torch.tensor([5.0, 0.0])
    forward[1, 2] = torch.tensor([0.0, 5.0])
    return forward


@pytest.fixture

def dummy_target_all_valid():
    return torch.zeros((2, 4), dtype=torch.long)


@pytest.fixture

def dummy_target_with_padding():
    return torch.tensor([
        [0, -100, 1, -100],
        [2, 3, -100, 4],
    ], dtype=torch.long)


@ pytest.mark.parametrize(
    "forward_fixture, per_sample_top_k, per_sample_ratio, expected_mask", [
        # Uniform logits -> arbitrary token order; pick first K per sample
        ("dummy_uniform_forward_output", 1, None, [1, 0, 0, 0, 1, 0, 0, 0]),
        ("dummy_uniform_forward_output", None, 50, [1, 1, 0, 0, 1, 1, 0, 0]),
        # Varied logits -> uniform tokens have highest entropy => positions where forward==0
        ("dummy_varied_forward_output", 2, None, [1, 0, 1, 0, 0, 1, 0, 1]),
        ("dummy_varied_forward_output", None, 50, [1, 0, 1, 0, 0, 1, 0, 1]),
    ],
)

def test_per_sample_selection(
    base_params,
    request,
    dummy_target_all_valid,
    forward_fixture,
    per_sample_top_k,
    per_sample_ratio,
    expected_mask,
):
    params = base_params.copy()
    if per_sample_top_k is not None:
        params["per_sample_top_k"] = per_sample_top_k
    if per_sample_ratio is not None:
        params["per_sample_ratio"] = per_sample_ratio

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
    forward = request.getfixturevalue(forward_fixture)
    sampler.inform_samples(
        sample_ids=[0, 1],
        forward_input=None,
        forward_output=forward,
        target=dummy_target_all_valid,
    )
    token_ids, weights = sampler.select_points()
    assert weights.tolist() == expected_mask
    assert int(weights.sum().item()) == sum(expected_mask)


def test_global_fallback(base_params, dummy_uniform_forward_output, dummy_target_all_valid):
    # No per-sample config -> use global downsampling_ratio
    sampler = RemoteTokenUncertaintyDownsampling(
        pipeline_id=0,
        trigger_id=0,
        batch_size=2,
        params_from_selector=base_params.copy(),
        modyn_config={},
        per_sample_loss=None,
        device="cpu",
    )
    sampler.init_downsampler()
    sampler.inform_samples(
        sample_ids=[0, 1],
        forward_input=None,
        forward_output=dummy_uniform_forward_output,
        target=dummy_target_all_valid,
    )
    _, weights = sampler.select_points()
    # global 50% of 8 tokens = 4
    assert int(weights.sum().item()) == 4


def test_invalid_per_sample_combination(base_params):
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


def test_padding_ignored(base_params, dummy_uniform_forward_output, dummy_target_with_padding):
    sampler = RemoteTokenUncertaintyDownsampling(
        pipeline_id=0,
        trigger_id=0,
        batch_size=2,
        params_from_selector=base_params.copy(),
        modyn_config={},
        per_sample_loss=None,
        device="cpu",
    )
    sampler.init_downsampler()
    sampler.inform_samples(
        sample_ids=[0, 1],
        forward_input=None,
        forward_output=dummy_uniform_forward_output,
        target=dummy_target_with_padding,
    )
    token_ids, weights = sampler.select_points()
    # Ensure padding tokens have weight=0
    for (s, idx), w in zip(token_ids, weights.tolist()):
        if dummy_target_with_padding[s, idx] == -100:
            assert w == 0
    # Ensure non-padding tokens count matches weights sum
    non_pad = (dummy_target_with_padding != -100).sum().item()
    assert sum(weights.tolist()) == min(non_pad, int(0.5 * non_pad))
