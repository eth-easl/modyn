import unittest
import sqlite3
import os
from unittest.mock import MagicMock

from mock import patch

from scorer import Scorer
from datastorage import DataStorage


class TestScorer(unittest.TestCase):
    STORAGE_LOCATION = os.getcwd()

    @patch.multiple(Scorer, __abstractmethods__=set())
    def setUp(self):
        def __init__(self, config):
            self._con = sqlite3.connect(':memory:')
            self._data_storage = DataStorage()

        with patch.object(Scorer, '__init__', __init__):
            self.scorer = Scorer(None)
            self.scorer.get_score = MagicMock(return_value=0)
            self.scorer.setup_database()

    def test_setup_database(self):
        cursor = self.scorer.get_con().cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        self.assertEqual(tables[0][0], 'batch_metadata')
        self.assertEqual(tables[1][0], 'row_metadata')

        cursor.execute("SELECT * FROM batch_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 5)
        cursor.execute("SELECT * FROM row_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 3)

    def test_add_batch_to_metadata(self):
        test_file = 'test_file1.csv'
        batch_id = self.scorer.add_batch_to_metadata(test_file)
        self.assertEqual(batch_id, 1)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[1], test_file)
        self.assertEqual(row[4], 1)

    def test_add_row_to_metadata(self):
        test_row = 42
        test_batch_id = 9
        test_score = 0.5
        self.scorer.add_row_to_metadata(test_row, test_batch_id, test_score)

        cursor = self.scorer._con.cursor()
        cursor.execute("SELECT * FROM row_metadata;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[0], test_row)
        self.assertEqual(row[1], test_batch_id)
        self.assertEqual(row[2], test_score)

    def test_add_batch(self):
        test_file = 'test_file1.csv'
        rows1 = [6, 7, 8]

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

    def test_fetch_batches(self):
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

        results = self.scorer.fetch_batches(self.scorer.BATCHES_BY_SCORE, 2)
        self.assertEqual(results[0][1], filename2)
        self.assertEqual(results[1][1], filename3)

        results = self.scorer.fetch_batches(
            self.scorer.BATCHES_BY_TIMESTAMP, 2)
        self.assertEqual(results[0][1], filename4)
        self.assertEqual(results[1][1], filename3)

    def test_fetch_rows(self):
        cursor = self.scorer._con.cursor()

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
            (13, 2, 0.9))

        results = self.scorer.fetch_rows(self.scorer.ROWS_BY_SCORE, 2, 1)
        self.assertEqual(results[0], 11)
        self.assertEqual(results[1], 10)

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
