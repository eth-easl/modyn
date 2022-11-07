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
            self._insert_batch_metadata_sql = '''INSERT INTO batch_metadata(timestamp) VALUES(?);'''
            self._insert_row_metadata_sql = '''INSERT INTO row_metadata(row, filename, batch_id, score)
                                                VALUES(?, ?, ?, ?);'''
            self._update_batch_metadata_sql = '''UPDATE batch_metadata SET score = ?, new = ? WHERE id = ?;'''
            self._update_row_metadata_sql = '''UPDATE row_metadata SET score = ? WHERE batch_id = ? AND row = ?;'''
            self._create_batch_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS batch_metadata (
                                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                        timestamp INTEGER,
                                                        score REAL,
                                                        new INTEGER DEFAULT 1);'''
            self._create_row_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS row_metadata (
                                                        row INTEGER,
                                                        batch_id INTEGER,
                                                        filename VARCHAR(100),
                                                        score REAL,
                                                        PRIMARY KEY (row, batch_id),
                                                        FOREIGN KEY (batch_id) REFERENCES batch_metadata(id));'''
            self._select_statement = 'SELECT filename, row FROM row_metadata WHERE batch_id=? ORDER BY row ASC'

        with patch.object(LastSelector, '__init__', __init__):
            self.scorer = LastSelector(None)
            self.scorer._setup_database()

    def test_get_next_batch(self):
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
            (10, filename4, 4, 0.5))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (11, filename4, 4, 0.8))
        cursor.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (12, filename4, 4, 0.3))
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

        result = self.scorer._get_next_batch()
        self.assertCountEqual(result['test4.json'], [10, 11, 12])

        cursor.execute(
            "SELECT new FROM batch_metadata WHERE id=4;")
        result_rows = cursor.fetchall()
        self.assertTrue(result_rows[0][0] == 0)
