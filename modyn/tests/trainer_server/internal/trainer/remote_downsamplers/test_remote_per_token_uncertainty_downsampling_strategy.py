# tests/test_remote_token_ds.py
import pytest, torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_token_uncertainty_downsampling import (
    RemoteTokenUncertaintyDownsampling,
)

# ────────────────────────────────────────────────────────────────────────
# shared fixtures
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def base_params():
    return dict(
        downsampling_ratio=50,  # %
        ratio_max=100,
        score_metric="Entropy",
        generative=True,
    )

# helper to create a down-sampler quickly
def make_ds(params, batch_size=2):
    return RemoteTokenUncertaintyDownsampling(
        0, 0, batch_size=batch_size,
        params_from_selector=params, modyn_config={},
        per_sample_loss=None, device="cpu"
    )

# ────────────────────────────────────────────────────────────────────────
# 1. Simple uniform-logit tests (ties everywhere)  ← from previous file
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def logits_uniform():
    # B=2, T=4, V=2  → all tokens have identical entropy
    return torch.zeros((2, 4, 2), dtype=torch.float32)

@pytest.fixture
def target_all_valid():
    return torch.zeros((2, 4), dtype=torch.long)  # no -100s

@pytest.mark.parametrize(
    "weight_ps, top_k, ratio, expected_mask",
    [
        #       ↓ token indices 0‥7
        # token-level, top-k=1          → keep 0,4
        (False, 1,   None, [1,0,0,0, 1,0,0,0]),
        # token-level, 50 %/sample      → keep 0,1,4,5
        (False, None, 50,  [1,1,0,0, 1,1,0,0]),
        # token-level, global-ratio=50 %→ keep first 4 tokens
        (False, None, None,[1,1,1,1, 0,0,0,0]),
        # sample-level, top-k=1         → keep *all* tokens of sample 0
        (True,  1,   None, [1,1,1,1, 0,0,0,0]),
        # sample-level, 50 %/samples    → same as above (1 of 2 samples)
        (True,  None, 50,  [1,1,1,1, 0,0,0,0]),
        # sample-level, global ratio    → same again
        (True,  None, None,[1,1,1,1, 0,0,0,0]),
    ],
)
def test_uniform_matrix(base_params, logits_uniform, target_all_valid,
                        weight_ps, top_k, ratio, expected_mask):
    params = dict(base_params, weight_per_sample=weight_ps)
    if top_k  is not None: params["per_sample_top_k"]  = top_k
    if ratio  is not None: params["per_sample_ratio"]  = ratio

    ds = make_ds(params)
    ds.init_downsampler()
    ds.inform_samples([0, 1], None, logits_uniform, target_all_valid)
    _, weights = ds.select_points()

    assert weights.tolist() == expected_mask
    assert int(weights.sum()) == sum(expected_mask)

# ────────────────────────────────────────────────────────────────────────
# 2.  Non-uniform logits  –  different entropies per token
# ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def logits_mixed():
    # B=2, T=3, V=3
    #
    # Sample 0
    #   token-0 → [0,0,0]  (flat ⇒ *high* entropy  )
    #   token-1 → [5,0,0]  (peaked⇒  low  entropy  )
    #   token-2 → [0,5,0]  (peaked⇒  low  entropy  )
    # Sample 1
    #   token-3 → [0,0,0]  (high entropy)
    #   token-4 → [0,0,5]  (low  entropy)
    #   token-5 → [0,0,0]  (high entropy)
    return torch.tensor(
        [[[0,0,0],[5,0,0],[0,5,0]],
         [[0,0,0],[0,0,5],[0,0,0]]],
        dtype=torch.float32
    )

@pytest.fixture
def target_mixed():
    # Mark sample-1 / token-5 as invalid (-100):
    return torch.tensor([[0,0,0],
                         [0,0,-100]], dtype=torch.long)

def test_token_topk_with_structure(base_params, logits_mixed, target_mixed):
    """
    token-level, per_sample_top_k=1
    Expect: pick the *most-uncertain* token per sample, ignoring invalid.
      · Sample 0 → token-0   (index 0)
      · Sample 1 → token-3   (index 3)
    """
    params = dict(base_params,
                  weight_per_sample=False,
                  per_sample_top_k=1)
    ds = make_ds(params, batch_size=2)
    ds.init_downsampler()
    ds.inform_samples([0,1], None, logits_mixed, target_mixed)
    _, w = ds.select_points()

    assert w.tolist() == [1,0,0, 1,0,0]  # length 6
    assert int(w.sum()) == 2

def test_sample_topk_with_invalid(base_params, logits_mixed, target_mixed):
    """
    sample-level, top_k=1
    · Sample 0 mean entropy ≈ -0.4
    · Sample 1 mean entropy ≈ -0.75 (higher uncertainty)
    → keep **sample 1**  → its *valid* tokens: indices 3 & 4
      (token-5 is invalid)
    """
    params = dict(base_params,
                  weight_per_sample=True,
                  per_sample_top_k=1)
    ds = make_ds(params, batch_size=2)
    ds.init_downsampler()
    ds.inform_samples([0,1], None, logits_mixed, target_mixed)
    _, w = ds.select_points()

    assert w.tolist() == [0,0,0, 1,1,0]  # only tokens 3 & 4
    assert int(w.sum()) == 2

def test_per_sample_ratio_with_different_lengths(base_params):
    """
    B=3, T=4
      · Sample 0 → 4 valid tokens
      · Sample 1 → 2 valid tokens
      · Sample 2 → 1 valid token
    ratio=50 % per sample, token-level
      ⇒ keep ceil(0.5·L) per sample
        → 2 + 1 + 1 = 4 tokens in total
    """
    logits = torch.zeros((3,4,2))
    target = torch.tensor([[0,0,0,0],
                           [0,0,-100,-100],
                           [0,-100,-100,-100]])
    params = dict(base_params,
                  weight_per_sample=False,
                  per_sample_ratio=50)
    ds = make_ds(params, batch_size=3)
    ds.init_downsampler()
    ds.inform_samples([0,1,2], None, logits, target)
    _, w = ds.select_points()
    assert int(w.sum()) == 4

# ────────────────────────────────────────────────────────────────────────
# 3. Misc edge cases
# ────────────────────────────────────────────────────────────────────────
def test_all_tokens_invalid_returns_all_zeros(base_params):
    logits = torch.randn((1,5,4))
    target = torch.full((1,5), -100)
    ds = make_ds(base_params, batch_size=1)
    ds.init_downsampler()
    ds.inform_samples([0], None, logits, target)
    _, w = ds.select_points()
    # all weights must be 0 because every token is invalid
    assert w.sum() == 0

def test_downsampling_ratio_greater_than_total_tokens(base_params, logits_uniform, target_all_valid):
    """
    downsampling_ratio=400 % while ratio_max=100
    → target = ceil(4·N) but capped by 'max(...)' logic to at least 1.
      Since ratio > 100 the implementation still uses the same formula,
      which in practice selects *all* valid tokens (can’t exceed N).
    """
    params = dict(base_params, downsampling_ratio=400)  # 400 %
    ds = make_ds(params)
    ds.init_downsampler()
    ds.inform_samples([0,1], None, logits_uniform, target_all_valid)
    _, w = ds.select_points()
    assert w.sum() == 8  # keeps every token

def test_mutually_exclusive_flags_still_fail(base_params):
    p = dict(base_params, per_sample_top_k=2, per_sample_ratio=10)
    with pytest.raises(ValueError):
        make_ds(p)
