
from unittest.mock import patch
from modyn.backend.selector.selector import Selector


def noop_constructor_mock(self):
    pass


@patch.multiple(Selector, __abstractmethods__=set())
@patch.object(Selector, '__init__', noop_constructor_mock)
@patch.object(Selector, '_create_or_fetch_existing_set')
@patch.object(Selector, '_get_info_for_training')
def test_get_sample_keys(test__get_info_for_training, test__create_or_fetch_existing_set):
    test__create_or_fetch_existing_set.return_value = ["a", "b", "c"]
    test__get_info_for_training.return_value = tuple([3, 3])

    selector = Selector()

    assert selector.get_sample_keys(0, 0, 0) == ["a"]
    assert selector.get_sample_keys(0, 0, 1) == ["b"]
    assert selector.get_sample_keys(0, 0, 2) == ["c"]

    # TODO(MaxiBoether): Assert throws with invalid worker id
