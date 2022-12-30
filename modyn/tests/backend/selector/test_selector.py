
from unittest.mock import patch
from modyn.backend.selector.selector import Selector
from modyn.backend.selector.custom_selectors.base_selector import BasicSelector
from modyn.backend.selector.custom_selectors.gdumb_selector import GDumbSelector
from modyn.backend.selector.custom_selectors.score_selector import ScoreSelector

from collections import Counter
import pytest

import numpy as np

# We do not use the parameters in this empty mock constructor.


def noop_constructor_mock(self, config: dict):  # pylint: disable=unused-argument
    pass


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(Selector, '__init__', noop_constructor_mock)
@patch.object(Selector, '_get_info_for_training')
def test_get_training_set_partition(test__get_info_for_training):
    test__get_info_for_training.return_value = tuple([10, 3])

    # We need to instantiate an abstract class for the test
    selector = Selector(None)  # pylint: disable=abstract-class-instantiated

    training_samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    assert selector._get_training_set_partition(0, training_samples, 0) == ['a', 'b', 'c', 'd']
    assert selector._get_training_set_partition(0, training_samples, 1) == ['e', 'f', 'g', 'h']
    assert selector._get_training_set_partition(0, training_samples, 2) == ['i', 'j']

    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, 3)
    with pytest.raises(Exception):
        selector._get_training_set_partition(0, training_samples, -1)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(Selector, '__init__', noop_constructor_mock)
@patch.object(Selector, '_prepare_training_set')
@patch.object(Selector, '_get_info_for_training')
def test_get_sample_keys(test__get_info_for_training, test__prepare_training_set):
    test__prepare_training_set.return_value = ["a", "b", "c"]
    test__get_info_for_training.return_value = tuple([3, 3])

    # We need to instantiate an abstract class for the test
    selector = Selector(None)  # pylint: disable=abstract-class-instantiated

    assert selector.get_sample_keys(0, 0, 0) == ["a"]
    assert selector.get_sample_keys(0, 0, 1) == ["b"]
    assert selector.get_sample_keys(0, 0, 2) == ["c"]
    with pytest.raises(Exception):
        selector.get_sample_keys(0, 0, -1)
    with pytest.raises(Exception):
        selector.get_sample_keys(0, 0, 3)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(Selector, '__init__', noop_constructor_mock)
@patch.object(Selector, '_register_training')
def test_register_training(test__register_training):
    # We need to instantiate an abstract class for the test
    selector = Selector(None)  # pylint: disable=abstract-class-instantiated
    test__register_training.return_value = 5

    assert selector.register_training(1000, 1) == 5
    with pytest.raises(Exception):
        selector.register_training(1000, 0)
    with pytest.raises(Exception):
        selector.register_training(0, 1)
    with pytest.raises(Exception):
        selector.register_training(-1000, 1)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BasicSelector, '__init__', noop_constructor_mock)
@patch.object(BasicSelector, 'get_unseen_data')
@patch.object(BasicSelector, 'get_seen_data')
def test_base_selector_get_new_training_samples(test_get_seen_data, test_get_unseen_data):
    test_get_unseen_data.return_value = ["a", "b", "c"]
    test_get_seen_data.return_value = ["d"]

    # We need to instantiate an abstract class for the test
    selector = BasicSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_unseen_data_ratio(0.75)
    selector._set_is_adaptive_ratio(False)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_unseen_data_ratio(-0.1)

    assert selector._select_new_training_samples(0, 4) == ["a", "b", "c", "d"]
    test_get_unseen_data.assert_called_with(0, 3)
    test_get_seen_data.assert_called_with(0, 1)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BasicSelector, '__init__', noop_constructor_mock)
@patch.object(BasicSelector, 'get_unseen_data')
@patch.object(BasicSelector, 'get_seen_data')
@patch.object(BasicSelector, 'get_unseen_data_size')
@patch.object(BasicSelector, 'get_seen_data_size')
def test_adaptive_selector_get_new_training_samples(test_get_seen_data_size,
                                                    test_get_unseen_data_size,
                                                    test_get_seen_data,
                                                    test_get_unseen_data):
    test_get_unseen_data.return_value = ["a"]
    test_get_seen_data.return_value = ["b", "c", "d", "e"]
    test_get_seen_data_size.return_value = 80
    test_get_unseen_data_size.return_value = 20

    # We need to instantiate an abstract class for the test
    selector = BasicSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_adaptive_ratio(True)

    assert selector._select_new_training_samples(0, 5) == ["a", "b", "c", "d", "e"]
    test_get_unseen_data.assert_called_with(0, 1)
    test_get_seen_data.assert_called_with(0, 4)

@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BasicSelector, '__init__', noop_constructor_mock)
@patch.object(BasicSelector, 'get_samples_by_metadata_query')
def test_base_selector_get_seen_data(test_get_samples_by_metadata_query):
    test_get_samples_by_metadata_query.return_value = ['a', 'b'], [0, 1], [1, 1], [0, 0], ['a', 'b']

    # We need to instantiate an abstract class for the test
    selector = BasicSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_adaptive_ratio(True)

    for key in selector.get_seen_data(0, 1):
        assert key in ['a', 'b'] 

    query = """SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 1 AND training_id = 0"""
    test_get_samples_by_metadata_query.assert_called_with(query)

    assert selector.get_seen_data_size(0) == 2
    test_get_samples_by_metadata_query.assert_called_with(query)

@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BasicSelector, '__init__', noop_constructor_mock)
@patch.object(BasicSelector, 'get_samples_by_metadata_query')
def test_base_selector_get_unseen_data(test_get_samples_by_metadata_query):
    test_get_samples_by_metadata_query.return_value = ['a', 'b'], [0, 1], [0, 0], [0, 0], ['a', 'b']

    # We need to instantiate an abstract class for the test
    selector = BasicSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_adaptive_ratio(True)

    for key in selector.get_unseen_data(0, 1):
        assert key in ['a', 'b'] 

    query = """SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 0 AND training_id = 0"""
    test_get_samples_by_metadata_query.assert_called_with(query)

    assert selector.get_unseen_data_size(0) == 2
    test_get_samples_by_metadata_query.assert_called_with(query)

@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(GDumbSelector, '__init__', noop_constructor_mock)
@patch.object(GDumbSelector, 'get_samples_by_metadata_query')
def test_gdumb_selector_get_metadata(test_get_samples_by_metadata_query):
    test_get_samples_by_metadata_query.return_value = ['a', 'b'], [0, 1], [0, 0], [0, 4], ['a', 'b']

    # We need to instantiate an abstract class for the test
    selector = GDumbSelector(None)  # pylint: disable=abstract-class-instantiated

    assert selector._get_all_metadata(0) == (['a', 'b'], [0, 4])

    query = f"SELECT key, score, seen, label, data FROM metadata_database WHERE training_id = 0"
    test_get_samples_by_metadata_query.assert_called_with(query)

@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(ScoreSelector, '__init__', noop_constructor_mock)
@patch.object(ScoreSelector, 'get_samples_by_metadata_query')
def test_score_selector_get_metadata(test_get_samples_by_metadata_query):
    test_get_samples_by_metadata_query.return_value = ['a', 'b'], [-1.5, 2.4], [0, 0], [0, 4], ['a', 'b']

    # We need to instantiate an abstract class for the test
    selector = ScoreSelector(None)  # pylint: disable=abstract-class-instantiated

    assert selector._get_all_metadata(0) == (['a', 'b'], [-1.5, 2.4])

    query = f"SELECT key, score, seen, label, data FROM metadata_database WHERE training_id = 0"
    test_get_samples_by_metadata_query.assert_called_with(query)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(GDumbSelector, '__init__', noop_constructor_mock)
@patch.object(GDumbSelector, '_get_all_metadata')
def test_gdumb_selector_get_new_training_samples(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]

    test__get_all_metadata.return_value = all_samples, all_classes

    # We need to instantiate an abstract class for the test
    selector = GDumbSelector(None)  # pylint: disable=abstract-class-instantiated

    samples = selector._select_new_training_samples(0, 6)
    classes = [clss for _, clss in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(classes) == Counter([1, 1, 2, 2, 3, 3])
    original_samples = set(zip(all_samples, all_classes))
    for sample, clss in zip(samples, classes):
        assert (sample, clss) in original_samples


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(ScoreSelector, '__init__', noop_constructor_mock)
@patch.object(ScoreSelector, '_get_all_metadata')
def test_score_selector_normal_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [1, 0, 1, 0, 1, 0, 1, 0]

    test__get_all_metadata.return_value = all_samples, all_scores

    # We need to instantiate an abstract class for the test
    selector = ScoreSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(False)

    samples = selector._select_new_training_samples(0, 4)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(scores) == Counter([0.25, 0.25, 0.25, 0.25])
    original_samples = set(zip(all_samples, all_scores))
    for sample, score in zip(samples, scores):
        assert (sample, score * 4) in original_samples


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(ScoreSelector, '__init__', noop_constructor_mock)
@patch.object(ScoreSelector, '_get_all_metadata')
def test_score_selector_softmax_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [10, -10, 10, -10, 10, -10, 10, -10]

    test__get_all_metadata.return_value = all_samples, all_scores

    # We need to instantiate an abstract class for the test
    selector = ScoreSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_softmax_mode(True)

    samples = selector._select_new_training_samples(0, 4)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert (np.array(scores) - 88105.8633608).max() < 1e4
    original_samples = set(all_samples)
    for sample in samples:
        assert sample in original_samples
