# pylint: disable=no-value-for-parameter
import os
import pathlib
from collections import Counter
from unittest.mock import patch

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.metadata_database.models.training import Training
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.backend.selector.internal.selector_strategies.gdumb_strategy import GDumbStrategy

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


def setup():
    with MetadataDatabaseConnection(get_minimal_modyn_config()) as database:
        database.create_tables()

        trainig = Training(1, 1)
        database.get_session().add(trainig)
        database.get_session().commit()

        metadata = Metadata("test_key", 0.5, False, 1, b"test_data", trainig.id)

        metadata.id = 1  # SQLite does not support autoincrement for composite primary keys
        database.get_session().add(metadata)

        metadata2 = Metadata("test_key2", 0.75, True, 2, b"test_data2", trainig.id)

        metadata2.id = 2  # SQLite does not support autoincrement for composite primary keys
        database.get_session().add(metadata2)

        database.get_session().commit()


def teardown():
    os.remove(database_path)


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(GDumbStrategy, "__init__", noop_constructor_mock)
def test_gdumb_selector_get_metadata():
    selector = GDumbStrategy(None)

    assert selector._get_all_metadata(1) == (["test_key", "test_key2"], [1, 2])


@patch.multiple(AbstractSelectionStrategy, __abstractmethods__=set())
@patch.object(GDumbStrategy, "__init__", noop_constructor_mock)
@patch.object(GDumbStrategy, "_get_all_metadata")
def test_gdumb_selector_get_new_training_samples(test__get_all_metadata):
    all_samples = ["a", "b", "c", "d", "e", "f", "g", "h"]
    all_classes = [1, 1, 1, 1, 2, 2, 3, 3]

    test__get_all_metadata.return_value = all_samples, all_classes

    selector = GDumbStrategy(None)

    samples = selector.select_new_training_samples(0, 6)
    classes = [clss for _, clss in samples]
    samples = [sample for sample, _ in samples]

    assert len(classes) == len(samples) == 6
    assert Counter(classes) == Counter([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for sample in samples:
        assert sample in all_samples
