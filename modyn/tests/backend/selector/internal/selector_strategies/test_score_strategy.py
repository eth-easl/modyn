# pylint: disable=no-value-for-parameter
import os
import pathlib
from collections import Counter
from unittest.mock import patch

import numpy as np
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.score_strategy import ScoreStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"


def get_minimal_modyn_config():
    return {
        "metadata_database": {
            "drivername": "sqlite",
            "username": "",
            "password": "",
            "host": "",
            "port": "0",
            "database": f"{database_path}",
        },
    }


def noop_constructor_mock(self, config=None, opt=None):  # pylint: disable=unused-argument
    self._modyn_config = get_minimal_modyn_config()
    with MetadataDatabaseConnection(self._modyn_config) as database:
        self.database = database


def setup():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        trainig = Training(1, 1)
        database.session.add(trainig)
        database.session.commit()

        metadata = Metadata("test_key", 0.5, False, 1, b"test_data", trainig.training_id)

        metadata.metadata_id = 1  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata)

        metadata2 = Metadata("test_key2", 0.75, True, 1, b"test_data", trainig.training_id)

        metadata2.metadata_id = 2  # SQLite does not support autoincrement for composite primary keys
        database.session.add(metadata2)

        database.session.commit()


def teardown():
    os.remove(database_path)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, "__init__", noop_constructor_mock)
def test_score_selector_get_metadata():
    selector = ScoreStrategy(None)

    assert selector._get_all_metadata(1) == (["test_key", "test_key2"], [0.5, 0.75])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(ScoreStrategy, "__init__", noop_constructor_mock)
@patch.object(ScoreStrategy, "_get_all_metadata")
def test_score_selector_normal_mode(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_scores = [1, 0, 1, 0, 1, 0, 1, 0]

    test__get_all_metadata.return_value = all_samples, all_scores

    selector = ScoreStrategy(None)
    selector._set_is_softmax_mode(False)

    samples = selector.select_new_training_samples(0, 4)
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

    selector = ScoreStrategy(None)
    selector._set_is_softmax_mode(True)

    samples = selector.select_new_training_samples(0, 4)
    scores = [score for _, score in samples]
    samples = [sample for sample, _ in samples]

    assert (np.array(scores) - 88105.8633608).max() < 1e4
    original_samples = set(all_samples)
    for sample in samples:
        assert sample in original_samples
