import unittest
import sqlite3
import pathlib 
import os 

from mock import patch
import pandas as pd

from datascorer import DataScorer
from datastorage import DataStorage

STORAGE_LOCATION = os.getcwd()

class TestDataScorer(unittest.TestCase):

    def setUp(self):
        def __init__(self, config):
            self.con = sqlite3.connect(':memory:')
            self.data_storage = DataStorage()

        with patch.object(DataScorer, '__init__', __init__):
            self.data_scorer = DataScorer(None)
            self.data_scorer.setup_database()

    def test_setup_database(self):
        con = sqlite3.connect('metadata.db')
        cursor = con.cursor()
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
        batch_id = self.data_scorer.add_batch_to_metadata(test_file)
        self.assertEqual(batch_id, 1)

        cursor = self.data_scorer.con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[1], test_file)
        self.assertEqual(row[4], 1)

    def test_add_row_to_metadata(self):
        test_row = 42
        test_batch_id = 9
        test_score = 0.5
        self.data_scorer.add_row_to_metadata(test_row, test_batch_id, test_score)

        cursor = self.data_scorer.con.cursor()
        cursor.execute("SELECT * FROM row_metadata;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[0], test_row)
        self.assertEqual(row[1], test_batch_id)
        self.assertEqual(row[2], test_score)

    def test_add_batch(self):
        test_file = 'test_file1.csv'
        rows1 = [6,7,8]

        self.data_scorer.add_batch(test_file, rows1)

        cursor = self.data_scorer.con.cursor()
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

        rows2 = [42,96,106]

        self.data_scorer.add_batch(test_file, rows2)

        cursor.execute("SELECT * FROM batch_metadata WHERE id=2;")
        row = cursor.fetchall()[0]
        batch_id = row[0]
        self.assertEqual(row[1], test_file)
        self.assertTrue(0 <= row[3] <= 1)
        self.assertEqual(row[4], 1)

        cursor.execute("SELECT * FROM row_metadata WHERE batch_id=2 ORDER BY row ASC;")
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

        cursor = self.data_scorer.con.cursor()

        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename1, 10.1, 0.3, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename2, 10.5, 0.8, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename3, 11, 0.7, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename4, 15, 0.5, 1))

        results = self.data_scorer.fetch_batches(self.data_scorer.BATCHES_BY_SCORE, 2)
        self.assertEqual(results[0][1], filename2)
        self.assertEqual(results[1][1], filename3)

        results = self.data_scorer.fetch_batches(self.data_scorer.BATCHES_BY_TIMESTAMP, 2)
        self.assertEqual(results[0][1], filename4)
        self.assertEqual(results[1][1], filename3)

    def test_fetch_rows(self):
        cursor = self.data_scorer.con.cursor()

        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                    (10, 1, 0.5))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                    (11, 1, 0.8))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                    (12, 1, 0.3))
        cursor.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                    (13, 2, 0.9))

        results = self.data_scorer.fetch_rows(self.data_scorer.ROWS_BY_SCORE, 2, 1)
        self.assertEqual(results[0], 11)
        self.assertEqual(results[1], 10)

    def test_create_shuffled_batches(self):
        batchname1 = 'test1'
        batchname2 = 'test2'
        batchname3 = 'test3'
        batchname4 = 'test4'

        df = pd.read_csv(STORAGE_LOCATION + '/tests/data/test.csv', header=0)
        df['row_id'] = [10, 11, 12]
        filename1 = self.data_scorer.data_storage.write_dataset_to_tar(batchname1, df.to_json())
        df['row_id'] = [20, 21, 22]
        filename2 = self.data_scorer.data_storage.write_dataset_to_tar(batchname2, df.to_json())
        df['row_id'] = [30, 31, 32]
        filename3 = self.data_scorer.data_storage.write_dataset_to_tar(batchname3, df.to_json())
        df['row_id'] = [40, 41, 42]
        filename4 = self.data_scorer.data_storage.write_dataset_to_tar(batchname4, df.to_json())

        cursor = self.data_scorer.con.cursor()

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

        self.data_scorer.create_shuffled_batches(self.data_scorer.BATCHES_BY_SCORE, self.data_scorer.ROWS_BY_SCORE, 2, 4)

        cursor.execute("SELECT * FROM row_metadata WHERE batch_id = 5;")
        row = cursor.fetchall()
        self.assertEqual(len(row), 4)

    def test_get_next_batch(self):
        filename1 = 'test1.tar'
        filename2 = 'test2.tar'
        filename3 = 'test3.tar'
        filename4 = 'test4.tar'

        cursor = self.data_scorer.con.cursor()

        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename1, 10.1, 0.3, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename2, 10.5, 0.8, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename3, 11, 0.7, 1))
        cursor.execute('''INSERT INTO batch_metadata(filename, timestamp, score, new) VALUES(?, ?, ?, ?)''',
                    (filename4, 15, 0.5, 1))

        result = self.data_scorer.get_next_batch()
        self.assertEqual(result, filename1)
