import pytest
import torch

from modyn.config.schema.pipeline import MeteorMetricConfig
from modyn.evaluator.internal.metrics import Meteor
from modyn.evaluator.internal.metrics.abstract_text_metric import AbstractTextMetric


@pytest.fixture
def meteor_metric():
    return Meteor(MeteorMetricConfig(), tokenizer="T5TokenizerTransform")


def test_meteor_type_and_name(meteor_metric):
    assert isinstance(meteor_metric, AbstractTextMetric)
    assert meteor_metric.get_name() == "METEOR"


def test_meteor_identical_sentences_tensor(meteor_metric):
    y_true = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    y_pred = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    meteor_metric._dataset_evaluated_callback(y_true, y_pred, 1)
    result = meteor_metric.get_evaluation_result()
    assert result > 0.98  # Exact match


def test_meteor_partial_sentences_tensor(meteor_metric):
    y_true = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 92, 7543]])
    y_pred = torch.tensor([[71, 598, 18345, 2210, 15, 45, 83, 997, 7543]])  # red → green
    meteor_metric._dataset_evaluated_callback(y_true, y_pred, 1)
    result = meteor_metric.get_evaluation_result()
    assert 0.0 < result < 1.0


def test_meteor_empty_input(meteor_metric):
    y_true = []
    y_pred = []
    meteor_metric._dataset_evaluated_callback(y_true, y_pred, 0)
    assert meteor_metric.get_evaluation_result() == 0


def test_meteor_mismatched_input_lengths(meteor_metric):
    y_true = ["A reference sentence."]
    y_pred = []
    meteor_metric._dataset_evaluated_callback(y_true, y_pred, 1)
    assert meteor_metric.get_evaluation_result() == 0
