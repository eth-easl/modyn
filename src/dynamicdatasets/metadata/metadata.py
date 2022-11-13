import time
from abc import ABC
import sqlite3

import psycopg2


class Metadata(ABC):

    _config = None
    _con = None
    _insert_training_set_metadata_sql = '''INSERT INTO training_set_metadata(timestamp) VALUES(%s) RETURNING id;'''
    _insert_sample_metadata_sql = '''INSERT INTO sample_metadata(sample, filename, score) VALUES(%s, %s, %s);'''
    _update_training_set_metadata_sql = '''UPDATE training_set_metadata SET score = %s, new = %s WHERE id = %s;'''
    _update_sample_metadata_sql = '''UPDATE sample_metadata
                                  SET score = %s
                                  WHERE sample = %s AND EXISTS (SELECT *
                                                            FROM training_set_to_sample
                                                            WHERE sample = sample_metadata.sample
                                                            AND training_set_id = %s);'''
    _get_scores_sql = '''SELECT score
                         FROM sample_metadata
                         JOIN training_set_to_sample ON training_set_to_sample.sample = sample_metadata.sample
                         WHERE training_set_id = %s;'''
    _select_sample_sql = '''SELECT * FROM sample_metadata WHERE sample = %s;'''
    _insert_training_set_to_sample_sql = '''INSERT INTO training_set_to_sample(training_set_id, sample)
                                            VALUES(%s, %s);'''
    _create_training_set_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS training_set_metadata (
            id SERIAL PRIMARY KEY,
            timestamp INTEGER,
            score REAL,
            new INTEGER DEFAULT 1);'''
    #  Rename sample to sample
    _create_sample_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS sample_metadata (
            sample SERIAL PRIMARY KEY,
            filename VARCHAR(100),
            score REAL,
            new INTEGER DEFAULT 1);'''
    _create_training_set_to_sample_table_sql = '''CREATE TABLE IF NOT EXISTS training_set_to_sample (
            id SERIAL PRIMARY KEY,
            training_set_id INTEGER,
            sample INTEGER,
            FOREIGN KEY (training_set_id) REFERENCES training_set_metadata(id),
            FOREIGN KEY (sample) REFERENCES sample_metadata(sample));'''
    # TODO: Fix this (psycopg2.errors.InvalidForeignKey: there is no unique
    # constraint matching given keys for referenced table "sample_metadata")

    def __init__(self, config: dict):
        self._config = config
        self._con = psycopg2.connect(
            host="db",
            user="postgres",
            password="postgres")
        self._setup_database()

    def _setup_database(self):
        cur = self._con.cursor()
        cur.execute(self._create_training_set_metadata_table_sql)
        cur.execute(self._create_sample_metadata_table_sql)
        cur.execute(self._create_training_set_to_sample_table_sql)
        self._con.commit()

    def _add_training_set_to_metadata(self) -> int:
        """
        Insert a training_set into the metadata database

        Returns:
            int: unique id for this training_set
        """
        cur = self._con.cursor()
        cur.execute(
            self._insert_training_set_metadata_sql,
            (time.time(),))
        try:
            training_set_id = cur.fetchone()[0]
        except TypeError:
            training_set_id = cur.lastrowid
        self._con.commit()
        return training_set_id

    def _add_sample_to_metadata(
            self,
            sample: int,
            training_set_id: int,
            score: float,
            filename: str):
        """
        Insert a data item from a training_set to the sample metadata database

        Args:
            sample (int): sample id to uniquely identify the sample
            training_set_id (int): id of the corresponding training_set
            score (float): initial score of the sample
        """
        cur = self._con.cursor()
        cur.execute(self._select_sample_sql, (sample,))
        if cur.fetchone() is None:
            cur.execute(
                self._insert_sample_metadata_sql,
                (sample, filename, score))
        cur.execute(self._insert_training_set_to_sample_sql,
                    (training_set_id, sample))
        self._con.commit()

    def _update_training_set_metadata(
            self,
            training_set_id: int,
            score: float,
            new: bool):
        """
        Update the metadata of a training_set

        Args:
            training_set_id (int): id of the corresponding training_set
            score (float): score to be updated
            new (bool): new flag to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            self._update_training_set_metadata_sql,
            (score, new, training_set_id))
        self._con.commit()

    def _update_sample_metadata(
            self,
            training_set_id: int,
            sample: int,
            score: float):
        """
        Update the metadata of a data item as a sample

        Args:
            training_set_id (int): id of the corresponding training_set
            sample (int): sample id to uniquely identify the sample
            score (float): score to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            self._update_sample_metadata_sql,
            (score, training_set_id, sample))
        self._con.commit()

    def _fetch_training_set_ids(
            self,
            sql_statment: str,
            training_set_count: int) -> list[int]:
        """
        Fetch the corresponding training_setes accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve training_setes
            training_set_count (int): number of training_setes to be retrieved

        Returns:
            list[int]: list of training_set ids
        """
        cur = self._con.cursor()
        cur.execute(sql_statment + str(training_set_count))
        training_setes = cur.fetchall()
        return [item for t in training_setes for item in t]

    def _get_scores(self, training_set_id) -> list[float]:
        """
        Get the scores of the samples in a training_set

        Args:
            training_set_id (int): training_set_id of the training_set

        Returns:
            list[float]: list of scores
        """
        cur = self._con.cursor()
        cur.execute(self._get_scores_sql, (training_set_id,))
        scores = cur.fetchall()
        return [item for t in scores for item in t]

    def _fetch_filenames_to_indexes(self,
                                    sql_statement: str,
                                    sample_count: int,
                                    training_set_id: int) -> dict[str,
                                                                  list[int]]:
        """
        Fetch the corresponding filename to indexes dictionary accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve samples
            sample_count (int): number of samples to be retrieved
            training_set_id (int): corresponding training_set id

        Returns:
            dict[str, list[int]]: map of filename to indexes of the samples in the training_set
        """
        cur = self._con.cursor()
        cur.execute(sql_statement + str(sample_count), (training_set_id,))
        samples = cur.fetchall()
        sample_dict = dict()
        for x, y in samples:
            sample_dict.setdefault(x, []).append(y)
        return sample_dict

    def get_con(self):
        return self._con
