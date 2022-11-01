import sqlite3
import logging
import time
import statistics
from abc import ABC, abstractmethod

from datastorage import DataStorage


class Scorer(ABC):
    """
    Abstract class to score the samples according to the implementors defined data importance measure 
    and update the corresponding metadata database
    """
    _config = None
    _data_storage = None
    _con = None
    _nr_batches_update = 3
    _batch_size = 100
    ROWS_BY_SCORE = "SELECT row, score from row_metadata WHERE batch_id=? ORDER BY score DESC LIMIT "
    BATCHES_BY_SCORE = "SELECT id, filename FROM batch_metadata ORDER BY score DESC LIMIT "
    BATCHES_BY_TIMESTAMP = "SELECT id, filename FROM batch_metadata ORDER BY timestamp DESC LIMIT "

    def __init__(self, config: dict, data_storage: DataStorage):
        """
        Args:
            config (dict): YAML config file with the required structure. 

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        self.config = config
        self.data_storage = data_storage
        self.nr_batches_update = config['data_scorer']['nr_files_update']
        self.batch_size = config['data_feeder']['batch_size']
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

    def add_batch(self, filename: str, rows: list[int], scores=None):
        """
        Add a batch to the data scorer metadata database

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch
        """
        logging.info(
            f'Adding batch from input file {filename} to metadata database')
        batch_id = self.add_batch_to_metadata(filename)
        scores = []
        for row in rows:
            score = self.get_score()
            self.add_row_to_metadata(row, batch_id, score)
            scores.append(score)
        median = statistics.median(scores)
        self.update_batch_metadata(batch_id, median, 1)

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

    def get_next_batch(self) -> str:
        """
        Get the next unread batch and update that it has been read

        Returns:
            str: filename of the next ready batch
        """
        cur = self._con.cursor()
        cur.execute(
            'SELECT id, filename, score FROM batch_metadata WHERE new=1 ORDER BY timestamp ASC LIMIT 1')
        row = cur.fetchall()[0]
        self.update_batch_metadata(row[0], row[2], 0)
        return row[1]

    @abstractmethod
    def create_shuffled_batches(
            self, batch_selection, row_selection, batch_count, batch_size):
        """
        Update an existing batch based on a batch and row selection criterion. Select a 
        number of batches and decide on a total number of samples for the new batch.
        The created batch will contain equal proportion of the selected batches.

        Args:
            batch_selection (str, optional): sql to select batches according to a criteria.
            row_selection (str, optional): sql to select rows accordig to a criteria.
            batch_count (int, optional): number of batches to include in the updated batch.
            batch_size (int, optional): number of rows in the resulting batch.
        """
        raise NotImplementedError

    @abstractmethod
    def get_score(self):
        """
        Generate score according to data importance

        Returns:
            float: score
        """
        raise NotImplementedError

    def get_con(self):
        return self._con
