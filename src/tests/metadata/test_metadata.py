import unittest
import sqlite3
import os
from unittest.mock import MagicMock
import pathlib
import shutil

from mock import patch

from dynamicdatasets.metadata import Metadata
from dynamicdatasets.offline.storage.storage import Storage

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


def __metadata_init__(self, config):
    self._con = sqlite3.connect(':memory:')
    self._config = {'metadata': {
        'nr_files_update': 3}, 'feeder': {'batch_size': 1}}
    self._data_storage = Storage(STORAGE_LOCATION + '/tests/tmp')
    self._insert_batch_metadata_sql = '''INSERT INTO batch_metadata(timestamp) VALUES(?);'''
    self._insert_row_metadata_sql = '''INSERT INTO row_metadata(row, filename, score)
                                            VALUES(?, ?, ?);'''
    self._update_batch_metadata_sql = '''UPDATE batch_metadata SET score = ?, new = ? WHERE id = ?;'''
    self._update_row_metadata_sql = '''UPDATE row_metadata
                                  SET score = ?
                                  WHERE row = ? AND EXISTS (SELECT *
                                                            FROM batch_to_row
                                                            WHERE row = row_metadata.row AND batch_id = ?);'''
    self._insert_batch_to_row_sql = '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?);'''
    self._create_batch_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS batch_metadata (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    timestamp INTEGER,
                                                    score REAL,
                                                    new INTEGER DEFAULT 1);'''
    self._create_row_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS row_metadata (
            row INTEGER PRIMARY KEY,
            filename VARCHAR(100),
            score REAL);'''
    self._create_batch_to_row_table_sql = '''CREATE TABLE IF NOT EXISTS batch_to_row (
            batch_id INTEGER,
            row INTEGER,
            PRIMARY KEY (batch_id, row),
            FOREIGN KEY (batch_id) REFERENCES batch_metadata(id),
            FOREIGN KEY (row) REFERENCES row_metadata(row));'''
    self._select_statement = '''SELECT filename, row_metadata.row
                                FROM row_metadata JOIN batch_to_row ON row_metadata.row = batch_to_row.row
                                WHERE batch_id=? ORDER BY row_metadata.row ASC'''
    self._row_selection = '''SELECT filename, row_metadata.row
                             FROM row_metadata JOIN batch_to_row ON batch_to_row.row = row_metadata.row
                             WHERE batch_id=? ORDER BY score DESC LIMIT '''
    self._get_scores_sql = '''SELECT score FROM row_metadata
                              JOIN batch_to_row ON batch_to_row.row = row_metadata.row WHERE batch_id = ?;'''
    self._select_row_sql = '''SELECT * FROM row_metadata WHERE row = ?;'''


class TestMetadata(unittest.TestCase):
    STORAGE_LOCATION = os.getcwd()
    ROWS_BY_SCORE = '''SELECT filename, row_metadata.row from row_metadata
                     JOIN batch_to_row ON row_metadata.row = batch_to_row.row
                     WHERE batch_id=? ORDER BY score DESC LIMIT '''
    BATCHES_BY_SCORE = "SELECT id FROM batch_metadata ORDER BY score DESC LIMIT "
    BATCHES_BY_TIMESTAMP = "SELECT id FROM batch_metadata ORDER BY timestamp DESC LIMIT "

    @patch.multiple(Metadata, __abstractmethods__=set())
    def setUp(self):
        with patch.object(Metadata, '__init__', __metadata_init__):
            self.metadata = Metadata(None)
            self.metadata.get_score = MagicMock(return_value=0)
            self.metadata._setup_database()

    def tearDown(self):
        shutil.rmtree(STORAGE_LOCATION + '/tests/tmp')

    def test_setup_database(self):
        cursor = self.metadata.get_con().cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        self.assertEqual(tables[0][0], 'batch_metadata')
        self.assertEqual(tables[2][0], 'row_metadata')

        cursor.execute("SELECT * FROM batch_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 4)
        cursor.execute("SELECT * FROM row_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 3)
        cursor.execute("SELECT * FROM batch_to_row;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 2)

    def test_add_batch_to_metadata(self):
        batch_id = self.metadata._add_batch_to_metadata()
        self.assertEqual(batch_id, 1)

        cursor = self.metadata._con.cursor()
        cursor.execute("SELECT * FROM batch_metadata;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[3], 1)

    def test_update_batch_metadata(self):
        cursor = self.metadata._con.cursor()

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

        self.metadata._update_batch_metadata(2, 0.9, 0)
        cursor.execute("SELECT * FROM batch_metadata WHERE id=2;")
        row = cursor.fetchall()[0]
        self.assertCountEqual(row, (2, 10.5, 0.9, 0))

    def test_update_row_metadata(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (1, 'test_file1.json', 0.3))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 1))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (2, 'test_file2.json', 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 2))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (3, 'test_file3.json', 0.7))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 3))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (4, 'test_file4.json', 0.5))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 4))

        self.metadata._update_row_metadata(1, 2, 0.9)
        cursor.execute("SELECT * FROM row_metadata WHERE row=2;")
        row = cursor.fetchall()[0]
        self.assertEqual(row[2], 0.8)

    def test_add_row_to_metadata(self):
        test_row = 42
        test_batch_id = 9
        test_score = 0.5
        test_filename = 'test_file1.json'
        self.metadata._add_row_to_metadata(
            test_row, test_batch_id, test_score, test_filename)

        cursor = self.metadata._con.cursor()
        cursor.execute("SELECT * FROM row_metadata;")
        row = cursor.fetchall()[0]
        self.assertCountEqual(row, (42, 'test_file1.json', 0.5))

    def test_fetch_batches(self):

        cursor = self.metadata._con.cursor()

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

        results = self.metadata._fetch_batch_ids(self.BATCHES_BY_SCORE, 2)
        self.assertCountEqual(results, [2, 3])

        results = self.metadata._fetch_batch_ids(
            self.BATCHES_BY_TIMESTAMP, 2)
        self.assertCountEqual(results, [4, 3])

    def test_fetch_filenames_to_indexes(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (10, 'test1', 0.5))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 10))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (11, 'test2', 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 11))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (12, 'test3', 0.3))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 12))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (13, 'test4', 0.9))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (2, 13))

        results = self.metadata._fetch_filenames_to_indexes(
            self.ROWS_BY_SCORE, 2, 1)
        self.assertEqual(results['test2'][0], 11)
        self.assertEqual(results['test1'][0], 10)

    def test_get_scores(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (10, 'test1', 0.5))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 10))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (11, 'test2', 0.8))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 11))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (12, 'test3', 0.3))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (1, 12))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, score) VALUES(?, ?, ?)''',
            (13, 'test4', 0.9))
        cursor.execute(
            '''INSERT INTO batch_to_row(batch_id, row) VALUES(?, ?)''', (2, 13))

        results = self.metadata._get_scores(1)
        self.assertCountEqual(results, [0.5, 0.8, 0.3])
