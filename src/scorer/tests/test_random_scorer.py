import sqlite3
import pathlib
from unittest.mock import MagicMock

from mock import patch
import pandas as pd

from . import TestScorer
from datastorage import DataStorage
from scorer import RandomScorer

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class TestRandomScorer(TestScorer):

    def setUp(self):
        def __init__(self, config):
            self._con = sqlite3.connect(':memory:')
            self._data_storage = DataStorage(STORAGE_LOCATION + '/scorer/tests/tmp')
            self._config = {'data_scorer': {'nr_files_update': 3}, 'data_feeder': {'batch_size': 1}}

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

        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (10, 1, 0.5))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (11, 1, 0.8))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (12, 1, 0.3))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (20, 2, 0.9))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (21, 2, 0.7))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (22, 2, 0.1))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (30, 3, 0.1))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (31, 3, 0.2))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (32, 3, 0.8))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (40, 4, 0.74))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (41, 4, 0.23))
        cursor.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (42, 4, 0.8))

        self.scorer.create_shuffled_batches(
            self.scorer.BATCHES_BY_SCORE, self.scorer.ROWS_BY_SCORE, 2, 4)

        cursor.execute("SELECT * FROM row_metadata WHERE batch_id = 5;")
        row = cursor.fetchall()
        print(row)
        self.assertEqual(len(row), 4)
        expected_results = [20,21,31,32]
        for idx, r in enumerate(row):
            self.assertEqual(r[0], expected_results[idx])

    def test_add_batch(self):
        test_file = 'test_file1.csv'
        rows1 = [6, 7, 8]

        self.scorer._data_storage.create_shuffled_batch = MagicMock()
        self.scorer._data_storage.create_shuffled_batch.return_value = (test_file,[1,2,3])
        self.scorer.add_batch(test_file, rows1)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        self.assertEqual(row[1], test_file)
        self.assertEqual(row[4], 1)

        cursor.execute("SELECT * FROM row_metadata ORDER BY row ASC;")
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows1[0])
        self.assertEqual(result_rows[0][1], batch_id)
        self.assertTrue(0 <= result_rows[0][2] <= 1)

        self.assertEqual(result_rows[2][0], rows1[2])
        self.assertEqual(result_rows[2][1], batch_id)
        self.assertTrue(0 <= result_rows[2][2] <= 1)

        rows2 = [42, 96, 106]

        self.scorer.add_batch(test_file, rows2)

        cursor.execute("SELECT * FROM batch_metadata WHERE id=2;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        self.assertEqual(row[1], test_file)
        self.assertTrue(0 <= row[3] <= 1)
        self.assertEqual(row[4], 1)

        cursor.execute(
            "SELECT * FROM row_metadata WHERE batch_id=2 ORDER BY row ASC;")
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows2[0])
        self.assertTrue(0 <= result_rows[0][2] <= 1)

        self.assertEqual(result_rows[2][0], rows2[2])
        self.assertTrue(0 <= result_rows[2][2] <= 1)

