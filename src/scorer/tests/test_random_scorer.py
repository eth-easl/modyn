import sqlite3

from mock import patch
import pandas as pd

from . import TestScorer
from datastorage import DataStorage
from scorer import RandomScorer


class TestRandomScorer(TestScorer):

    def setUp(self):
        def __init__(self, config):
            self._con = sqlite3.connect(':memory:')
            self._data_storage = DataStorage()

        with patch.object(RandomScorer, '__init__', __init__):
            self.scorer = RandomScorer(None)
            self.scorer.setup_database()

    def test_create_shuffled_batches(self):
        batchname1 = 'test1'
        batchname2 = 'test2'
        batchname3 = 'test3'
        batchname4 = 'test4'

        df = pd.read_csv(self.STORAGE_LOCATION +
                         '/scorer/tests/data/test.csv', header=0)
        df['row_id'] = [10, 11, 12]
        filename1 = self.scorer._data_storage.write_dataset_to_tar(
            batchname1, df.to_json())
        df['row_id'] = [20, 21, 22]
        filename2 = self.scorer._data_storage.write_dataset_to_tar(
            batchname2, df.to_json())
        df['row_id'] = [30, 31, 32]
        filename3 = self.scorer._data_storage.write_dataset_to_tar(
            batchname3, df.to_json())
        df['row_id'] = [40, 41, 42]
        filename4 = self.scorer._data_storage.write_dataset_to_tar(
            batchname4, df.to_json())

        cursor = self.scorer._con.cursor()

        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                       (filename1, 10.1, 0.3, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                       (filename2, 10.5, 0.8, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                       (filename3, 11, 0.7, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                       (filename4, 15, 0.5, 1))

        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (10, 1, 0.5))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (11, 1, 0.8))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (12, 1, 0.3))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (20, 2, 0.9))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (21, 2, 0.7))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (22, 2, 0.1))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (30, 3, 0.1))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (31, 3, 0.2))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (32, 3, 0.8))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (40, 4, 0.74))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (41, 4, 0.23))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                       (42, 4, 0.8))

        self.scorer.create_shuffled_batches(
            self.scorer.BATCHES_BY_SCORE, self.scorer.ROWS_BY_SCORE, 2, 4)

        cursor.execute("SELECT * FROM row_metadata WHERE batch_id = 5;")
        row = cursor.fetchall()
        self.assertEqual(len(row), 4)
