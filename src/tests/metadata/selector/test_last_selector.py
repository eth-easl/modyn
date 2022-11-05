import unittest
import sqlite3
from unittest.mock import MagicMock
import pathlib
from mock import patch

from dynamicdatasets.metadata.selector import LastSelector
from dynamicdatasets.offline.storage import Storage

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class TestLastSelector(unittest.TestCase):

    @patch.multiple(LastSelector, __abstractmethods__=set())
    def setUp(self):
        def __init__(self, config):
            self._con = sqlite3.connect(':memory:')
            self._data_storage = Storage(
                STORAGE_LOCATION + '/scorer/tests/tmp')

        with patch.object(LastSelector, '__init__', __init__):
            self.scorer = LastSelector(None)
            self.scorer.setup_database()

    def test_get_next_batch(self):
        filename1 = 'test1.tar'
        filename2 = 'test2.tar'
        filename3 = 'test3.tar'
        filename4 = 'test4.tar'

        cursor = self.scorer._con.cursor()

        cursor.execute(
            '''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
            (filename1, 10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
            (filename2, 10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
            (filename3, 11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
            (filename4, 15, 0.5, 1))

        result = self.scorer.get_next_batch()
        self.assertEqual(result, filename1)

        cursor.execute(
            "SELECT new FROM batch_metadata WHERE filename='test1.tar';")
        result_rows = cursor.fetchall()
        self.assertTrue(result_rows[0][0] == 0)
