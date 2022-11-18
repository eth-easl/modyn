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
        'nr_files_update': 3}, 'feeder': {'training_set_size': 1}}
    self._data_storage = Storage(STORAGE_LOCATION + '/tests/tmp')
    self._insert_training_set_metadata_sql = '''INSERT INTO training_set_metadata(timestamp) VALUES(?);'''
    self._insert_sample_metadata_sql = '''INSERT INTO sample_metadata(sample, filename, score)
                                            VALUES(?, ?, ?);'''
    self._update_training_set_metadata_sql = '''UPDATE training_set_metadata SET score = ?, new = ? WHERE id = ?;'''
    self._update_sample_metadata_sql = '''UPDATE sample_metadata
                                  SET score = ?
                                  WHERE sample = ? AND EXISTS (SELECT *
                                                            FROM training_set_to_sample
                                                            WHERE sample = sample_metadata.sample
                                                            AND training_set_id = ?);'''
    self._insert_training_set_to_sample_sql = '''INSERT INTO training_set_to_sample(training_set_id, sample)
                                                 VALUES(?, ?);'''
    self._create_training_set_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS training_set_metadata (
                                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                    timestamp INTEGER,
                                                    score REAL,
                                                    new INTEGER DEFAULT 1);'''
    self._create_sample_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS sample_metadata (
            sample INTEGER PRIMARY KEY,
            filename VARCHAR(100),
            score REAL);'''
    self._create_training_set_to_sample_table_sql = '''CREATE TABLE IF NOT EXISTS training_set_to_sample (
            training_set_id INTEGER,
            sample INTEGER,
            PRIMARY KEY (training_set_id, sample),
            FOREIGN KEY (training_set_id) REFERENCES training_set_metadata(id),
            FOREIGN KEY (sample) REFERENCES sample_metadata(sample));'''

    self._select_statement = '''SELECT filename, sample_metadata.sample
                                FROM sample_metadata JOIN training_set_to_sample
                                ON sample_metadata.sample = training_set_to_sample.sample
                                WHERE training_set_id=? ORDER BY sample_metadata.sample ASC'''
    self._sample_selection = '''SELECT filename, sample_metadata.sample
                             FROM sample_metadata JOIN training_set_to_sample
                             ON training_set_to_sample.sample = sample_metadata.sample
                             WHERE training_set_id=? ORDER BY score DESC LIMIT '''
    self._get_scores_sql = '''SELECT score FROM sample_metadata
                              JOIN training_set_to_sample
                              ON training_set_to_sample.sample = sample_metadata.sample
                              WHERE training_set_id = ?;'''
    self._select_sample_sql = '''SELECT * FROM sample_metadata WHERE sample = ?;'''


class TestMetadata(unittest.TestCase):
    STORAGE_LOCATION = os.getcwd()
    SAMPLES_BY_SCORE = '''SELECT filename, sample_metadata.sample from sample_metadata
                     JOIN training_set_to_sample ON sample_metadata.sample = training_set_to_sample.sample
                     WHERE training_set_id=? ORDER BY score DESC LIMIT '''
    TRAINING_SETES_BY_SCORE = "SELECT id FROM training_set_metadata ORDER BY score DESC LIMIT "
    TRAINING_SETES_BY_TIMESTAMP = "SELECT id FROM training_set_metadata ORDER BY timestamp DESC LIMIT "

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
        self.assertEqual(tables[0][0], 'training_set_metadata')
        self.assertEqual(tables[2][0], 'sample_metadata')

        cursor.execute("SELECT * FROM training_set_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 4)
        cursor.execute("SELECT * FROM sample_metadata;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 3)
        cursor.execute("SELECT * FROM training_set_to_sample;")
        names = [description[0] for description in cursor.description]
        self.assertEqual(len(names), 2)

    def test_add_training_set_to_metadata(self):
        training_set_id = self.metadata._add_training_set_to_metadata()
        self.assertEqual(training_set_id, 1)

        cursor = self.metadata._con.cursor()
        cursor.execute("SELECT * FROM training_set_metadata;")
        sample = cursor.fetchall()[0]
        self.assertEqual(sample[3], 1)

    def test_update_training_set_metadata(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (15, 0.5, 1))

        self.metadata._update_training_set_metadata(2, 0.9, 0)
        cursor.execute("SELECT * FROM training_set_metadata WHERE id=2;")
        sample = cursor.fetchall()[0]
        self.assertCountEqual(sample, (2, 10.5, 0.9, 0))

    def test_update_sample_metadata(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (1, 'test_file1.json', 0.3))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             1))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (2, 'test_file2.json', 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             2))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (3, 'test_file3.json', 0.7))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             3))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (4, 'test_file4.json', 0.5))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             4))

        self.metadata._update_sample_metadata(1, 2, 0.9)
        cursor.execute("SELECT * FROM sample_metadata WHERE sample=2;")
        sample = cursor.fetchall()[0]
        self.assertEqual(sample[2], 0.8)

    def test_add_sample_to_metadata(self):
        test_sample = 42
        test_training_set_id = 9
        test_score = 0.5
        test_filename = 'test_file1.json'
        self.metadata._add_sample_to_metadata(
            test_sample, test_training_set_id, test_score, test_filename)

        cursor = self.metadata._con.cursor()
        cursor.execute("SELECT * FROM sample_metadata;")
        sample = cursor.fetchall()[0]
        self.assertCountEqual(sample, (42, 'test_file1.json', 0.5))

    def test_fetch_training_setes(self):

        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.1, 0.3, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (10.5, 0.8, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (11, 0.7, 1))
        cursor.execute(
            '''INSERT INTO training_set_metadata(timestamp, score, new) VALUES(?, ?, ?)''',
            (15, 0.5, 1))

        results = self.metadata._fetch_training_set_ids(
            self.TRAINING_SETES_BY_SCORE, 2)
        self.assertCountEqual(results, [2, 3])

        results = self.metadata._fetch_training_set_ids(
            self.TRAINING_SETES_BY_TIMESTAMP, 2)
        self.assertCountEqual(results, [4, 3])

    def test_fetch_filenames_to_indexes(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (10, 'test1', 0.5))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             10))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (11, 'test2', 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             11))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (12, 'test3', 0.3))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             12))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (13, 'test4', 0.9))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (2,
             13))

        results = self.metadata._fetch_filenames_to_indexes(
            self.SAMPLES_BY_SCORE, 2, 1)
        self.assertEqual(results['test2'][0], 11)
        self.assertEqual(results['test1'][0], 10)

    def test_get_scores(self):
        cursor = self.metadata._con.cursor()

        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (10, 'test1', 0.5))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             10))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (11, 'test2', 0.8))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             11))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (12, 'test3', 0.3))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (1,
             12))
        cursor.execute(
            '''INSERT INTO sample_metadata(sample, filename, score) VALUES(?, ?, ?)''',
            (13, 'test4', 0.9))
        cursor.execute(
            '''INSERT INTO training_set_to_sample(training_set_id, sample) VALUES(?, ?)''',
            (2,
             13))

        results = self.metadata._get_scores(1)
        self.assertCountEqual(results, [0.5, 0.8, 0.3])
