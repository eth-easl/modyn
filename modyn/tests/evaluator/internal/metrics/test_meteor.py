import pytest

from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics import AbstractHolisticMetric, Meteor


@pytest.fixture
def meteor_metric():
    """Fixture for initializing the METEOR metric."""
    return Meteor(MeteorMetricConfig())


def test_meteor_metric():
    """Test METEOR metric initialization."""
    meteor = Meteor(MeteorMetricConfig())
    assert isinstance(meteor, AbstractHolisticMetric)


def test_meteor_basic_evaluation(meteor_metric):
    """Test METEOR computation with valid text inputs."""
    y_true = ["The cat is on the mat.", "The dog ran away."]
    y_pred = ["A cat sits on the mat.", "The dog disappeared."]

    meteor_metric._dataset_evaluated_callback(y_true, y_pred, len(y_true))

    result = meteor_metric.get_evaluation_result()
    assert result > 0  # METEOR should be non-zero for similar sentences


def test_meteor_identical_text(meteor_metric):
    """Test METEOR for identical reference and prediction (should be 1.0)."""
    y_true = ["This is an exact match.", "Another perfect match."]
    y_pred = ["This is an exact match.", "Another perfect match."]

    meteor_metric._dataset_evaluated_callback(y_true, y_pred, len(y_true))

    result = meteor_metric.get_evaluation_result()
    assert result > 0.98  # Perfect match should give METEOR = 1.0


def test_meteor_empty_inputs(meteor_metric):
    """Test edge case with empty reference and prediction (should return 0)."""
    y_true = []
    y_pred = []

    meteor_metric._dataset_evaluated_callback(y_true, y_pred, 0)

    assert meteor_metric.get_evaluation_result() == 0


def test_meteor_invalid_inputs(meteor_metric):
    """Test METEOR with mismatched input lengths (should return 0)."""
    y_true = ["Valid reference."]
    y_pred = []  # Missing predictions

    meteor_metric._dataset_evaluated_callback(y_true, y_pred, len(y_true))

    assert meteor_metric.get_evaluation_result() == 0  # Expecting a 0 score
