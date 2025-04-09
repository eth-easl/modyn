import pytest
import torch

from modyn.config.schema.pipeline import PerplexityMetricConfig
from modyn.evaluator.internal.metrics import AbstractDecomposableMetric, Perplexity


@pytest.fixture
def perplexity_metric():
    """Fixture for initializing the Perplexity metric."""
    return Perplexity(PerplexityMetricConfig())


def test_perplexity_metric():
    """Test Perplexity initialization."""
    perplexity = Perplexity(PerplexityMetricConfig())
    assert isinstance(perplexity, AbstractDecomposableMetric)


def test_perplexity_evaluation(perplexity_metric):
    """Test Perplexity computation with valid logits."""
    y_true = torch.tensor([1, 2, 0]).unsqueeze(0)  # Target token IDs
    y_pred = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5], [0.4, 0.3, 0.3]]).unsqueeze(0)

    perplexity_metric._batch_evaluated_callback(y_true, y_pred, batch_size=1)

    result = perplexity_metric.get_evaluation_result()
    assert result > 0  # Perplexity should be positive


def test_perplexity_empty(perplexity_metric):
    """Test edge case where no data is seen (should return inf)."""
    assert perplexity_metric.get_evaluation_result() == float("inf")


def test_perplexity_invalid_shape(perplexity_metric):
    """Test invalid input shapes for logits."""
    y_true = torch.tensor([1, 2, 0])
    y_pred = torch.tensor([0.1, 0.7, 0.2])  # Invalid shape (missing logits dimension)

    with pytest.raises(RuntimeError):
        perplexity_metric._batch_evaluated_callback(y_true, y_pred, batch_size=1)
