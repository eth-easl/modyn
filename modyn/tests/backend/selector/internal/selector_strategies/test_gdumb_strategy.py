# pylint: disable=no-value-for-parameter
from collections import Counter
from unittest.mock import patch

from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.gdumb_strategy import GDumbStrategy


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
@patch.object(GDumbStrategy, "__init__", noop_constructor_mock)
def test_gdumb_selector_get_metadata():
    test_metadata_response = ["a", "b"], [0, 1], [0, 0], [0, 4], ["a", "b"]

    selector = GDumbStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector._grpc = MockGRPCHandler(test_metadata_response)

    assert selector._get_all_metadata(0) == (["a", "b"], [0, 4])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(GDumbStrategy, "__init__", noop_constructor_mock)
@patch.object(GDumbStrategy, "_get_all_metadata")
def test_gdumb_selector_get_new_training_samples(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]

    test__get_all_metadata.return_value = all_samples, all_classes

    selector = GDumbStrategy(None)  # pylint: disable=abstract-class-instantiated
    selector.training_set_size_limit = 6

    samples = selector.select_new_training_samples(0)
    classes = [clss for _, clss in samples]
    samples = [sample for sample, _ in samples]

    assert len(classes) == len(samples) == 6
    assert Counter(classes) == Counter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for sample in samples:
        assert sample in all_samples
