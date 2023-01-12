from unittest.mock import patch
from modyn.backend.selector.selector_strategy import SelectorStrategy
from modyn.backend.selector.internal.selector_strategies.data_freshness_strategy import DataFreshnessStrategy
from modyn.backend.selector.internal.selector_strategies.gdumb_strategy import GDumbStrategy
from modyn.backend.selector.internal.selector_strategies.score_strategy import ScoreStrategy

from collections import Counter
import pytest

import numpy as np


class MockGRPCHandler:

    def __init__(self, metadata_response):
        self.metadata_response = metadata_response

    def register_training(self, training_set_size, num_workers):  # pylint: disable=unused-argument
        return 5

    def get_samples_by_metadata_query(self, query):  # pylint: disable=unused-argument
        return self.metadata_response

    def get_info_for_training(self, training_id):
        return tuple([10, 3])

# We do not use the parameters in this empty mock constructor.


def noop_constructor_mock(self, config: dict):  # pylint: disable=unused-argument
    pass


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(SelectorStrategy, '__init__', noop_constructor_mock)
@patch.object(SelectorStrategy, '_select_new_training_samples')
def test_prepare_training_set(test__select_new_training_samples):
    test__select_new_training_samples.return_value = ['a', 'b']

    selector = SelectorStrategy(None)  # pylint: disable=abstract-class-instantiated
    assert selector._prepare_training_set(0, 0, 0) == ['a', 'b']

    test__select_new_training_samples.return_value = []
    with pytest.raises(ValueError):
        selector._prepare_training_set(0, 0, 3)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(SelectorStrategy, '__init__', noop_constructor_mock)
def test_get_training_set_partition():
    selector = SelectorStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.grpc = MockGRPCHandler(None)

    training_samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    assert selector._get_training_set_partition(0, training_samples, 0) == ['a', 'b', 'c', 'd']
    assert selector._get_training_set_partition(0, training_samples, 1) == ['e', 'f', 'g', 'h']
    assert selector._get_training_set_partition(0, training_samples, 2) == ['i', 'j']

    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, 3)
    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, -1)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(SelectorStrategy, '__init__', noop_constructor_mock)
@patch.object(SelectorStrategy, '_prepare_training_set')
# @patch.object(SelectorStrategy, '_get_info_for_training')
def test_get_sample_keys(test__prepare_training_set):
    training_samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    test__prepare_training_set.return_value = training_samples
    # test__get_info_for_training.return_value = tuple([3, 3])

    selector = SelectorStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.grpc = MockGRPCHandler(None)

    assert selector.get_sample_keys(0, 0, 0) == ["a", "b", "c", "d"]
    assert selector.get_sample_keys(0, 0, 1) == ["e", "f", "g", "h"]
    assert selector.get_sample_keys(0, 0, 2) == ["i", "j"]
    with pytest.raises(ValueError):
        selector.get_sample_keys(0, 0, -1)
    with pytest.raises(ValueError):
        selector.get_sample_keys(0, 0, 10)
    with pytest.raises(NotImplementedError):
        selector._select_new_training_samples(0, 0)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(SelectorStrategy, '__init__', noop_constructor_mock)
def test_register_training():
    selector = SelectorStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.grpc = MockGRPCHandler(None)

    assert selector.register_training(1000, 1) == 5
    with pytest.raises(Exception):
        selector.register_training(1000, 0)
    with pytest.raises(Exception):
        selector.register_training(0, 1)
    with pytest.raises(Exception):
        selector.register_training(-1000, 1)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, '__init__', noop_constructor_mock)
@patch.object(DataFreshnessStrategy, '_get_unseen_data')
@patch.object(DataFreshnessStrategy, '_get_seen_data')
def test_base_selector_get_new_training_samples(test__get_seen_data, test__get_unseen_data):
    test__get_unseen_data.return_value = ["a", "b", "c"]
    test__get_seen_data.return_value = ["d"]

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_unseen_data_ratio(0.75)
    selector._is_adaptive_ratio = False
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(-0.1)

    assert selector._select_new_training_samples(0, 4) == [("a",), ("b",), ("c",), ("d",)]
    test__get_unseen_data.assert_called_with(0, 3)
    test__get_seen_data.assert_called_with(0, 1)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, '__init__', noop_constructor_mock)
@patch.object(DataFreshnessStrategy, '_get_unseen_data')
@patch.object(DataFreshnessStrategy, '_get_seen_data')
@patch.object(DataFreshnessStrategy, '_get_unseen_data_size')
@patch.object(DataFreshnessStrategy, '_get_seen_data_size')
def test_adaptive_selector_get_new_training_samples(test__get_seen_data_size,
                                                    test__get_unseen_data_size,
                                                    test__get_seen_data,
                                                    test__get_unseen_data):
    test__get_unseen_data.return_value = ["a"]
    test__get_seen_data.return_value = ["b", "c", "d", "e"]
    test__get_seen_data_size.return_value = 80
    test__get_unseen_data_size.return_value = 20

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector.unseen_data_ratio = 0.0

    assert selector._select_new_training_samples(0, 5) == [("a",), ("b",), ("c",), ("d",), ("e",)]
    test__get_unseen_data.assert_called_with(0, 1)
    test__get_seen_data.assert_called_with(0, 4)


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, '__init__', noop_constructor_mock)
def test_base_selector_get_seen_data():
    test_metadata_response = ['a', 'b'], [0, 1], [1, 1], [0, 0], ['a', 'b']

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector.grpc = MockGRPCHandler(test_metadata_response)

    for key in selector._get_seen_data(0, 1):
        assert key in ['a', 'b']

    assert selector._get_seen_data_size(0) == 2


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(DataFreshnessStrategy, '__init__', noop_constructor_mock)
def test_base_selector_get_unseen_data():
    test_metadata_response = ['a', 'b'], [0, 1], [0, 0], [0, 0], ['a', 'b']

    selector = DataFreshnessStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._is_adaptive_ratio = True
    selector.grpc = MockGRPCHandler(test_metadata_response)

    for key in selector._get_unseen_data(0, 1):
        assert key in ['a', 'b']

    assert selector._get_unseen_data_size(0) == 2


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(GDumbStrategy, '__init__', noop_constructor_mock)
def test_gdumb_selector_get_metadata():
    test_metadata_response = ['a', 'b'], [0, 1], [0, 0], [0, 4], ['a', 'b']

    selector = GDumbStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.grpc = MockGRPCHandler(test_metadata_response)

    assert selector._get_all_metadata(0) == (['a', 'b'], [0, 4])


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, '__init__', noop_constructor_mock)
def test_score_selector_get_metadata():
    test_metadata_response = ['a', 'b'], [-1.5, 2.4], [0, 0], [0, 4], ['a', 'b']

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.grpc = MockGRPCHandler(test_metadata_response)

    assert selector._get_all_metadata(0) == (['a', 'b'], [-1.5, 2.4])


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(GDumbStrategy, '__init__', noop_constructor_mock)
@patch.object(GDumbStrategy, '_get_all_metadata')
def test_gdumb_selector_get_new_training_samples(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]

    test__get_all_metadata.return_value = all_samples, all_classes

    selector = GDumbStrategy(None)  # pylint: disable=abstract-class-instantiated

    samples = selector._select_new_training_samples(0, 6)
    classes = [clss for _, clss in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(classes) == Counter([1, 1, 2, 2, 3, 3])
    original_samples = set(zip(all_samples, all_classes))
    for sample, clss in zip(samples, classes):
        assert (sample, clss) in original_samples


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, '__init__', noop_constructor_mock)
@patch.object(ScoreStrategy, '_get_all_metadata')
def test_score_selector_normal_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [1, 0, 1, 0, 1, 0, 1, 0]

    test__get_all_metadata.return_value = all_samples, all_scores

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(False)

    samples = selector._select_new_training_samples(0, 4)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(scores) == Counter([0.25, 0.25, 0.25, 0.25])
    original_samples = set(zip(all_samples, all_scores))
    for sample, score in zip(samples, scores):
        assert (sample, score * 4) in original_samples


@patch.multiple(SelectorStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, '__init__', noop_constructor_mock)
@patch.object(ScoreStrategy, '_get_all_metadata')
def test_score_selector_softmax_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [10, -10, 10, -10, 10, -10, 10, -10]

    test__get_all_metadata.return_value = all_samples, all_scores

    selector = ScoreStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(True)

    samples = selector._select_new_training_samples(0, 4)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert (np.array(scores) - 88105.8633608).max() < 1e4
    original_samples = set(all_samples)
    for sample in samples:
        assert sample in original_samples
