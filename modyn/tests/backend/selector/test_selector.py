
from unittest.mock import patch
from modyn.backend.selector.selector import Selector
from modyn.backend.selector.base_selector import BaseSelector
from modyn.backend.selector.gdumb_selector import GDumbSelector

from collections import Counter
import pytest

# We do not use the parameters in this empty mock constructor.


def noop_constructor_mock(self, config: dict):  # pylint: disable=unused-argument
    pass


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(Selector, '__init__', noop_constructor_mock)
@patch.object(Selector, '_create_or_fetch_existing_set')
@patch.object(Selector, '_get_info_for_training')
def test_get_sample_keys(test__get_info_for_training, test__create_or_fetch_existing_set):
    test__create_or_fetch_existing_set.return_value = ["a", "b", "c"]
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
@patch.object(BaseSelector, '__init__', noop_constructor_mock)
@patch.object(Selector, 'get_from_newqueue')
@patch.object(Selector, 'get_from_odm')
def test_base_selector_get_new_training_samples(test_get_from_odm, test_get_from_newqueue):
    test_get_from_newqueue.return_value = ["a", "b", "c"]
    test_get_from_odm.return_value = ["d"]

    # We need to instantiate an abstract class for the test
    selector = BaseSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_new_data_ratio(0.75)
    selector._set_is_adaptive_ratio(False)
    with pytest.raises(Exception):
        selector._set_new_data_ratio(1.1)
    with pytest.raises(Exception):
        selector._set_new_data_ratio(-0.1)

    assert selector._select_new_training_samples(0, 4) == ["a", "b", "c", "d"]
    test_get_from_newqueue.assert_called_with(0, 3)
    test_get_from_odm.assert_called_with(0, 1)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BaseSelector, '__init__', noop_constructor_mock)
@patch.object(Selector, 'get_from_newqueue')
@patch.object(Selector, 'get_from_odm')
@patch.object(Selector, 'get_newqueue_size')
@patch.object(Selector, 'get_odm_size')
def test_adaptive_selector_get_new_training_samples(test_get_odm_size,
                                                    test_get_newqueue_size,
                                                    test_get_from_odm,
                                                    test_get_from_newqueue):
    test_get_from_newqueue.return_value = ["a"]
    test_get_from_odm.return_value = ["b", "c", "d", "e"]
    test_get_odm_size.return_value = 80
    test_get_newqueue_size.return_value = 20

    # We need to instantiate an abstract class for the test
    selector = BaseSelector(None)  # pylint: disable=abstract-class-instantiated
    selector._set_is_adaptive_ratio(True)

    assert selector._select_new_training_samples(0, 5) == ["a", "b", "c", "d", "e"]
    test_get_from_newqueue.assert_called_with(0, 1)
    test_get_from_odm.assert_called_with(0, 4)


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(BaseSelector, '__init__', noop_constructor_mock)
@patch.object(GDumbSelector, '_get_all_odm')
def test_gdumb_selector_get_new_training_samples(test__get_all_odm):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]

    test__get_all_odm.return_value = all_samples, all_classes

    # We need to instantiate an abstract class for the test
    selector = GDumbSelector(None)  # pylint: disable=abstract-class-instantiated

    samples = selector._select_new_training_samples(0, 6)
    classes = [clss for _, clss in samples]
    samples = [sample for sample, _ in samples]

    assert Counter(classes) == Counter([1, 1, 2, 2, 3, 3])
    original_samples = set(zip(all_samples, all_classes))
    for sample, clss in zip(samples, classes):
        assert (sample, clss) in original_samples
