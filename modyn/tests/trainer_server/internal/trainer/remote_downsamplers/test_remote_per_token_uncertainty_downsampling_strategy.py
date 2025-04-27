# tests/test_remote_token_ds.py
import pytest, torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_token_uncertainty_downsampling import (
    RemoteTokenUncertaintyDownsampling,
)

# ────────────────────────────────────────────────────────────────────────
# helpers & shared fixtures
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def base_params():
    return dict(
        downsampling_ratio=50,   # %
        ratio_max=100,
        score_metric="Entropy",
        generative=True,
    )


def make_ds(params, batch_size: int = 2):
    """Instantiate a RemoteTokenUncertaintyDownsampling quickly."""
    return RemoteTokenUncertaintyDownsampling(
        pipeline_id=0,
        trigger_id=0,
        batch_size=batch_size,
        params_from_selector=params,
        modyn_config={},
        per_sample_loss=None,
        device="cpu",
    )


def _assert_indexes_per_sample(n_samples: int, idx: list[int]) -> None:
    """Utility: sanity-check that returned sample-IDs are valid & unique."""
    assert len(idx) == len(set(idx)), "duplicate sample-IDs returned"
    assert set(idx).issubset(set(range(n_samples))), "invalid sample-ID returned"


# ────────────────────────────────────────────────────────────────────────
# 1. Uniform-logit tests (ties everywhere)
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def logits_uniform():
    # B=2, T=4, V=2 → every token has identical entropy
    return torch.zeros((2, 4, 2), dtype=torch.float32)


@pytest.fixture
def target_all_valid():
    return torch.zeros((2, 4), dtype=torch.long)  # no −100s (all tokens valid)


@pytest.mark.parametrize(
    "weight_ps, top_k, ratio, expected_mask, expected_indexes",
    [
        # token-level, top-k = 1 → one kept token per sample ⇒ samples 0 & 1 survive
        (False, 1,   None, [1,0,0,0, 1,0,0,0], [0, 1]),
        # token-level, 50 % / sample → samples 0 & 1 survive
        (False, None, 50,  [1,1,0,0, 1,1,0,0], [0, 1]),
        # token-level, global 50 % → only sample-0 has any kept tokens
        (False, None, None,[1,1,1,1, 0,0,0,0], [0]),
        # sample-level, top-k = 1 → keep *all* tokens of sample-0
        (True,  1,   None,[1,1,1,1, 0,0,0,0], [0]),
        # sample-level, 50 % / samples → same as above
        (True,  None, 50, [1,1,1,1, 0,0,0,0], [0]),
        # sample-level, global ratio 50 % → same again
        (True,  None, None,[1,1,1,1, 0,0,0,0], [0]),
    ],
)
def test_uniform_matrix(base_params, logits_uniform, target_all_valid,
                        weight_ps, top_k, ratio, expected_mask, expected_indexes):
    params = dict(base_params, weight_per_sample=weight_ps)
    if top_k  is not None:
        params["per_sample_top_k"] = top_k
    if ratio  is not None:
        params["per_sample_ratio"] = ratio

    ds = make_ds(params)
    ds.init_downsampler()
    ds.inform_samples([0, 1], None, logits_uniform, target_all_valid)
    selected_idx, weights = ds.select_points()

    assert selected_idx == expected_indexes
    assert weights.tolist() == expected_mask
    assert int(weights.sum()) == sum(expected_mask)


# ────────────────────────────────────────────────────────────────────────
# 2. Mixed-entropy logits – different entropies per token
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def logits_mixed():
    # B=2, T=3, V=3 (see doc-string comments in original tests)
    return torch.tensor(
        [[[0,0,0],[5,0,0],[0,5,0]],
         [[0,0,0],[0,0,5],[0,0,0]]],
        dtype=torch.float32,
    )


@pytest.fixture
def target_mixed():
    return torch.tensor([[0,0,0],
                         [0,0,-100]], dtype=torch.long)  # last token of sample-1 invalid


def test_token_topk_with_structure(base_params, logits_mixed, target_mixed):
    params = dict(base_params, weight_per_sample=False, per_sample_top_k=1)
    ds = make_ds(params, batch_size=2)
    ds.init_downsampler()
    ds.inform_samples([0, 1], None, logits_mixed, target_mixed)
    selected_idx, w = ds.select_points()

    assert selected_idx == [0, 1]          # one token kept per sample
    _assert_indexes_per_sample(2, selected_idx)
    assert w.tolist() == [1,0,0, 1,0,0]
    assert int(w.sum()) == 2


def test_sample_topk_with_invalid(base_params, logits_mixed, target_mixed):
    params = dict(base_params, weight_per_sample=True, per_sample_top_k=1)
    ds = make_ds(params, batch_size=2)
    ds.init_downsampler()
    ds.inform_samples([0, 1], None, logits_mixed, target_mixed)
    selected_idx, w = ds.select_points()

    assert selected_idx == [1]             # sample-1 has higher uncertainty
    _assert_indexes_per_sample(2, selected_idx)
    assert w.tolist() == [0,0,0, 1,1,0]
    assert int(w.sum()) == 2


def test_per_sample_ratio_with_different_lengths(base_params):
    """Three samples with unequal #valid tokens; token-level ratio=50 %."""
    logits = torch.zeros((3,4,2))
    target = torch.tensor([[0,0,0,0],
                           [0,0,-100,-100],
                           [0,-100,-100,-100]])
    params = dict(base_params, weight_per_sample=False, per_sample_ratio=50)
    ds = make_ds(params, batch_size=3)
    ds.init_downsampler()
    ds.inform_samples([0, 1, 2], None, logits, target)
    selected_idx, w = ds.select_points()

    assert selected_idx == [0, 1, 2]       # all samples contribute ≥1 kept token
    _assert_indexes_per_sample(3, selected_idx)
    assert int(w.sum()) == 4


# ────────────────────────────────────────────────────────────────────────
# 3. Misc / edge-case tests  (now check selected_idx too)
# ────────────────────────────────────────────────────────────────────────
def test_all_tokens_invalid_returns_all_zeros(base_params):
    logits  = torch.randn((1,5,4))
    target  = torch.full((1,5), -100)         # every token invalid
    ds = make_ds(base_params, batch_size=1)
    ds.init_downsampler()
    ds.inform_samples([0], None, logits, target)
    selected_idx, w = ds.select_points()

    assert selected_idx == []                # no sample qualifies
    _assert_indexes_per_sample(1, selected_idx)  # still a valid subset
    assert w.sum() == 0


def test_downsampling_ratio_greater_than_total_tokens(base_params,
                                                      logits_uniform,
                                                      target_all_valid):
    """downsampling_ratio > 100 % ⇒ implementation keeps *all* tokens."""
    params = dict(base_params, downsampling_ratio=400)  # 400 %
    ds = make_ds(params)
    ds.init_downsampler()
    ds.inform_samples([0, 1], None, logits_uniform, target_all_valid)
    selected_idx, w = ds.select_points()

    assert selected_idx == [0, 1]
    _assert_indexes_per_sample(2, selected_idx)
    assert w.sum() == 8                       # keeps every token


def test_mutually_exclusive_flags_still_fail(base_params):
    p = dict(base_params, per_sample_top_k=2, per_sample_ratio=10)
    with pytest.raises(ValueError):
        make_ds(p)
