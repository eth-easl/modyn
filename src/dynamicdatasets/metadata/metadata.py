import time
from abc import ABC
import sqlite3

import psycopg2


class Metadata(ABC):

    _config = None
    _con = None
    _insert_batch_metadata_sql = '''INSERT INTO batch_metadata(timestamp) VALUES(%s) RETURNING id;'''
    _insert_row_metadata_sql = '''INSERT INTO row_metadata(row, filename, score) VALUES(%s, %s, %s);'''
    _update_batch_metadata_sql = '''UPDATE batch_metadata SET score = %s, new = %s WHERE id = %s;'''
    _update_row_metadata_sql = '''UPDATE row_metadata
                                  SET score = %s
                                  WHERE row = %s AND EXISTS (SELECT *
                                                            FROM batch_to_row
                                                            WHERE row = row_metadata.row AND batch_id = %s);'''
    _get_scores_sql = '''SELECT score
                         FROM row_metadata
                         JOIN batch_to_row ON batch_to_row.row = row_metadata.row
                         WHERE batch_id = %s;'''
    _select_row_sql = '''SELECT * FROM row_metadata WHERE row = %s;'''
    _insert_batch_to_row_sql = '''INSERT INTO batch_to_row(batch_id, row) VALUES(%s, %s);'''
    _create_batch_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS batch_metadata (
            id SERIAL PRIMARY KEY,
            timestamp INTEGER,
            score REAL,
            new INTEGER DEFAULT 1);'''
    _create_row_metadata_table_sql = '''CREATE TABLE IF NOT EXISTS row_metadata (
            row INTEGER PRIMARY KEY,
            filename VARCHAR(100),
            score REAL);'''
    _create_batch_to_row_table_sql = '''CREATE TABLE IF NOT EXISTS batch_to_row (
            batch_id INTEGER,
            row INTEGER,
            PRIMARY KEY (batch_id, row),
            FOREIGN KEY (batch_id) REFERENCES batch_metadata(id),
            FOREIGN KEY (row) REFERENCES row_metadata(row));'''

    def __init__(self, config: dict):
        self._config = config
        self._con = psycopg2.connect(
            host="db",
            user="postgres",
            password="postgres")
        self._setup_database()

    def _setup_database(self):
        cur = self._con.cursor()
        cur.execute(self._create_batch_metadata_table_sql)
        cur.execute(self._create_row_metadata_table_sql)
        cur.execute(self._create_batch_to_row_table_sql)
        self._con.commit()

    def _add_batch_to_metadata(self) -> int:
        """
        Insert a batch into the metadata database

        Returns:
            int: unique id for this batch
        """
        cur = self._con.cursor()
        cur.execute(
            self._insert_batch_metadata_sql,
            (time.time(),))
        try:
            batch_id = cur.fetchone()[0]
        except TypeError:
            batch_id = cur.lastrowid
        self._con.commit()
        return batch_id

    def _add_row_to_metadata(
            self,
            row: int,
            batch_id: int,
            score: float,
            filename: str):
        """
        Insert a data item from a batch to the row metadata database

        Args:
            row (int): row id to uniquely identify the row
            batch_id (int): id of the corresponding batch
            score (float): initial score of the row
        """
        cur = self._con.cursor()
        cur.execute(self._select_row_sql, (row,))
        if cur.fetchone() is None:
            cur.execute(
                self._insert_row_metadata_sql,
                (row, filename, score))
        cur.execute(self._insert_batch_to_row_sql, (batch_id, row))
        self._con.commit()

    def _update_batch_metadata(self, batch_id: int, score: float, new: bool):
        """
        Update the metadata of a batch

        Args:
            batch_id (int): id of the corresponding batch
            score (float): score to be updated
            new (bool): new flag to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            self._update_batch_metadata_sql,
            (score, new, batch_id))
        self._con.commit()

    def _update_row_metadata(self, batch_id: int, row: int, score: float):
        """
        Update the metadata of a data item as a row

        Args:
            batch_id (int): id of the corresponding batch
            row (int): row id to uniquely identify the row
            score (float): score to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            self._update_row_metadata_sql,
            (score, batch_id, row))
        self._con.commit()

    def _fetch_batch_ids(
            self,
            sql_statment: str,
            batch_count: int) -> list[int]:
        """
        Fetch the corresponding batches accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve batches
            batch_count (int): number of batches to be retrieved

        Returns:
            list[int]: list of batch ids
        """
        cur = self._con.cursor()
        cur.execute(sql_statment + str(batch_count))
        batches = cur.fetchall()
        return [item for t in batches for item in t]

    def _get_scores(self, batch_id) -> list[float]:
        """
        Get the scores of the rows in a batch

        Args:
            batch_id (int): batch_id of the batch

        Returns:
            list[float]: list of scores
        """
        cur = self._con.cursor()
        cur.execute(self._get_scores_sql, (batch_id,))
        scores = cur.fetchall()
        return [item for t in scores for item in t]

    def _fetch_filenames_to_indexes(self, sql_statement: str, row_count: int,
                                    batch_id: int) -> dict[str, list[int]]:
        """
        Fetch the corresponding filename to indexes dictionary accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve rows
            row_count (int): number of rows to be retrieved
            batch_id (int): corresponding batch id

        Returns:
            dict[str, list[int]]: map of filename to indexes of the rows in the batch
        """
        cur = self._con.cursor()
        cur.execute(sql_statement + str(row_count), (batch_id,))
        rows = cur.fetchall()
        row_dict = dict()
        for x, y in rows:
            row_dict.setdefault(x, []).append(y)
        return row_dict

    def get_con(self):
        return self._con
