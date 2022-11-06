import sqlite3
import pathlib
from unittest.mock import MagicMock
import unittest
from mock import patch

from dynamicdatasets.metadata.scorer import RandomScorer

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class TestRandomScorer(unittest.TestCase):

    def setUp(self):
        def __init__(self, config):
            self._con = sqlite3.connect(':memory:')
            self._config = {'data_scorer': {
                'nr_files_update': 3}, 'data_feeder': {'batch_size': 1}}

        with patch.object(RandomScorer, '__init__', __init__):
            self.scorer = RandomScorer(None)
            self.scorer._setup_database()

    def test_create_shuffled_batches(self):
        filename1 = 'test1.json'
        filename2 = 'test2.json'
        filename3 = 'test3.json'
        filename4 = 'test4.json'

        cursor = self.scorer._con.cursor()

        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO batch_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (15, 0.5, 1))

        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (10, filename1, 1, 0.5))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (11, filename1, 1, 0.8))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (12, filename1, 1, 0.3))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (20, filename2, 2, 0.9))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (21, filename2, 2, 0.7))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (22, filename2, 2, 0.1))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (30, filename3, 3, 0.1))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (31, filename3, 3, 0.2))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (32, filename3, 3, 0.8))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (40, filename4, 4, 0.74))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (41, filename4, 4, 0.23))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (42, filename4, 4, 0.8))

        self.scorer._create_shuffled_batches(2, 4)

        cursor.execute("SELECT * FROM row_metadata WHERE batch_id = 5;")
        row = cursor.fetchall()
        self.assertEqual(len(row), 4)
        expected_results = [20, 21, 31, 32]
        for idx, r in enumerate(row):
            self.assertIn(r[0], expected_results)

    def test_add_batch(self):
        test_file = 'test_file1.csv'
        rows1 = [6, 7, 8]

        self.scorer._add_batch(test_file, rows1)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        cursor.execute(
            "SELECT * FROM row_metadata WHERE batch_id = ?;", (batch_id,))
        rows = cursor.fetchall()
        self.assertEqual(rows[0][2], test_file)
        self.assertEqual(row[3], 1)

        cursor.execute("SELECT * FROM row_metadata ORDER BY row ASC;")
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows1[0])
        self.assertEqual(result_rows[0][1], batch_id)
        self.assertTrue(0 <= result_rows[0][3] <= 1)

        self.assertEqual(result_rows[2][0], rows1[2])
        self.assertEqual(result_rows[2][1], batch_id)
        self.assertTrue(0 <= result_rows[2][3] <= 1)

        rows2 = [42, 96, 106]

        self.scorer._add_batch(test_file, rows2)

        cursor.execute("SELECT * FROM batch_metadata WHERE id=2;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        cursor.execute(
            "SELECT * FROM row_metadata WHERE batch_id = ?;", (batch_id,))
        rows = cursor.fetchall()
        self.assertEqual(rows[0][2], test_file)
        self.assertTrue(0 <= row[2] <= 1)
        self.assertEqual(row[3], 1)

        cursor.execute(
            "SELECT * FROM row_metadata WHERE batch_id=2 ORDER BY row ASC;")
        result_rows = cursor.fetchall()
        self.assertEqual(result_rows[0][0], rows2[0])
        self.assertTrue(0 <= result_rows[0][3] <= 1)

        self.assertEqual(result_rows[2][0], rows2[2])
        self.assertTrue(0 <= result_rows[2][3] <= 1)

    def test_get_cumulative_score(self):
        result = self.scorer._get_cumulative_score([1, 0.5, 0.5, 0])
        self.assertEqual(result, 0.5)
