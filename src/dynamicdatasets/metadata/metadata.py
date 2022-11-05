import sqlite3
import time

from abc import ABC


class Metadata(ABC):

    _config = None
    _con = None

    def __init__(self, config: dict):
        self._config = config
        self._con = sqlite3.connect(
            config['data_scorer']['in_memory_database'])
        self.setup_database()

    def setup_database(self):
        cur = self._con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS batch_metadata (
            id INTEGER PRIMARY KEY,
            filename VARCHAR(100),
            timestamp INTEGER,
            score REAL,
            new INTEGER NOT NULL);'''
                    )
        cur.execute('''CREATE TABLE IF NOT EXISTS row_metadata (
            row INTEGER,
            batch_id INTEGER,
            score REAL,
            PRIMARY KEY (row, batch_id),
            FOREIGN KEY (batch_id) REFERENCES batch_metadata(id));'''
                    )
        self._con.commit()

    def add_batch_to_metadata(self, filename: str) -> int:
        """
        Insert a batch into the metadata database

        Warning: filename could lead to a sql injection

        Args:
            filename (str): filename of the batch

        Returns:
            int: unique id for this batch
        """
        cur = self._con.cursor()
        cur.execute(
            '''INSERT INTO batch_metadata(filename, timestamp, new) VALUES(?, ?, ?)''',
            (filename, time.time(),
             1))
        batch_id = cur.lastrowid
        self._con.commit()
        return batch_id

    def add_row_to_metadata(self, row: int, batch_id: int, score: float):
        """
        Insert a data item from a batch to the row metadata database

        Args:
            row (int): row id to uniquely identify the row
            batch_id (int): id of the corresponding batch
            score (float): initial score of the row
        """
        cur = self._con.cursor()
        cur.execute(
            '''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
            (row, batch_id, score))
        self._con.commit()

    def update_batch_metadata(self, batch_id: int, score: float, new: bool):
        """
        Update the metadata of a batch

        Args:
            batch_id (int): id of the corresponding batch
            score (float): score to be updated
            new (bool): new flag to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            '''UPDATE batch_metadata SET score = ?, new = ? WHERE id = ?''',
            (score, new, batch_id))
        self._con.commit()

    def update_row_metadata(self, batch_id: int, row: int, score: float):
        """
        Update the metadata of a data item as a row

        Args:
            batch_id (int): id of the corresponding batch
            row (int): row id to uniquely identify the row
            score (float): score to be updated
        """
        cur = self._con.cursor()
        cur.execute(
            '''UPDATE row_metadata SET score = ? WHERE batch_id = ? AND row = ?''',
            (score, batch_id, row))
        self._con.commit()

    def fetch_batches(self, sql_statment: str, batch_count: int) -> list[tuple[int, str]]:
        """
        Fetch the corresponding batches accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve batches
            batch_count (int): number of batches to be retrieved

        Returns:
            list[tuple[int, str]]: list of tuples of batch ids to filename
        """
        cur = self._con.cursor()
        cur.execute(sql_statment + str(batch_count))
        batches = cur.fetchall()
        return batches

    def fetch_rows(self, sql_statement: str, row_count: int, batch_id: int) -> list[int]:
        """
        Fetch the corresponding rows accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve rows
            row_count (int): number of rows to be retrieved
            batch_id (int): corresponding batch id

        Returns:
            list[int]: list of int of row ids
        """
        cur = self._con.cursor()
        cur.execute(sql_statement + str(row_count), (batch_id,))
        rows = cur.fetchall()
        return [i[0] for i in rows]

    def get_con(self):
        return self._con
