# pylint: disable=no-value-for-parameter
from collections import Counter
from unittest.mock import patch

import numpy as np
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.score_strategy import ScoreStrategy


class MockGRPCHandler:
    def __init__(self, metadata_response):
        self.metadata_response = metadata_response

    def register_training(self, training_set_size, num_workers):  # pylint: disable=unused-argument
        return 5

    def get_samples_by_metadata_query(self, query):  # pylint: disable=unused-argument
        return self.metadata_response

    def get_info_for_training(self, training_id):  # pylint: disable=unused-argument
        return 3


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    pass


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, "__init__", noop_constructor_mock)
def test_score_selector_get_metadata():
    test_metadata_response = ["a", "b"], [-1.5, 2.4], [0, 0], [0, 4], ["a", "b"]

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._grpc = MockGRPCHandler(test_metadata_response)

    assert selector._get_all_metadata(0) == (["a", "b"], [-1.5, 2.4])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, "__init__", noop_constructor_mock)
@patch.object(ScoreStrategy, "_get_all_metadata")
def test_score_selector_normal_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [1, 0, 1, 0, 1, 0, 1, 0]

    test__get_all_metadata.return_value = all_samples, all_scores

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(False)
    selector.training_set_size_limit = 4

    samples = selector.select_new_training_samples(0)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(scores) == Counter([0.25, 0.25, 0.25, 0.25])
    original_samples = set(zip(all_samples, all_scores))
    for sample, score in zip(samples, scores):
        assert (sample, score * 4) in original_samples


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, "__init__", noop_constructor_mock)
@patch.object(ScoreStrategy, "_get_all_metadata")
def test_score_selector_softmax_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [10, -10, 10, -10, 10, -10, 10, -10]

    test__get_all_metadata.return_value = all_samples, all_scores

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(True)
    selector.training_set_size_limit = 4

    samples = selector.select_new_training_samples(0)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert (np.array(scores) - 88105.8633608).max() < 1e4
    original_samples = set(all_samples)
    for sample in samples:
        assert sample in original_samples
