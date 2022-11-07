import time
from abc import ABC

import psycopg2


class Metadata(ABC):

    _config = None
    _con = None

    def __init__(self, config: dict):
        self._config = config
        self._con = psycopg2.connect(
            host="db",
            user="postgres",
            password="postgres")
        self._setup_database()

    def _setup_database(self):
        cur = self._con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS batch_metadata (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            score REAL,
            new INTEGER DEFAULT 1);'''
                    )
        cur.execute('''CREATE TABLE IF NOT EXISTS row_metadata (
            row INTEGER,
            batch_id INTEGER,
            filename VARCHAR(100),
            score REAL,
            PRIMARY KEY (row, batch_id),
            FOREIGN KEY (batch_id) REFERENCES batch_metadata(id));'''
                    )
        self._con.commit()

    def _add_batch_to_metadata(self) -> int:
        """
        Insert a batch into the metadata database

        Returns:
            int: unique id for this batch
        """
        cur = self._con.cursor()
        cur.execute(
            '''INSERT INTO batch_metadata(timestamp) VALUES(?)''',
            (time.time(),))
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
        cur.execute(
            '''INSERT INTO row_metadata(row, filename, batch_id, score) VALUES(?, ?, ?, ?)''',
            (row, filename, batch_id, score))
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
            '''UPDATE batch_metadata SET score = ?, new = ? WHERE id = ?''',
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
            '''UPDATE row_metadata SET score = ? WHERE batch_id = ? AND row = ?''',
            (score, batch_id, row))
        self._con.commit()

    def fetch_batches(self, sql_statment: str, batch_count: int) -> list[int]:
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

    def _fetch_rows(self, sql_statement: str, row_count: int,
                    batch_id: int) -> dict[str, list[int]]:
        """
        Fetch the corresponding rows accoring to the sql statement

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
