import pathlib
import json
import time
import sqlite3
import random
import statistics
import logging

from datastorage import DataStorage

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())
logger = logging.getLogger('DataOrchestrator')
handler = logging.FileHandler('DataOrchestrator.log')
logger.addHandler(handler)


class DataOrchestrator:
    config = None
    data_storage = None
    con = None
    ROWS_BY_SCORE = "SELECT row from row_metadata WHERE batch_id=? ORDER BY score DESC LIMIT ?"
    BATCHES_BY_SCORE = "SELECT id, filename FROM batch_metadata ORDER BY score DESC LIMIT ?"
    BATCHES_BY_TIMESTAMP = "SELECT id, filename FROM batch_metadata ORDER BY timestamp DESC LIMIT 2"

    def __init__(self, config: dict, data_storage: DataStorage):
        self.config = config
        self.data_storage = data_storage
        self.con = sqlite3.connect(
            config['data_orchestrator']['in_memory_database'])
        self.setup_database()

    def setup_database(self):
        logger.info('Setting up database')
        cur = self.con.cursor()
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
        self.con.commit()

    def add_batch(self, filename: str, rows: list[int]):
        """
        Add a batch to the data orchestrator metadata database

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch
        """
        logging.info(f'Adding batch from input file {filename} to metadata database')
        batch_id = self.add_batch_to_metadata(filename)
        scores = []
        for row in rows:
            score = self.get_initial_random_score()
            self.add_row_to_metadata(row, batch_id, score)
            scores.append(score)
        median = statistics.median(scores)
        self.update_batch_metadata(batch_id, median, 1)

    def add_batch_to_metadata(self, filename: str) -> int:
        """
        Insert a batch into the metadata database

        Args:
            filename (str): filename of the batch

        Returns:
            int: unique id for this batch
        """
        cur = self.con.cursor()
        cur.execute('''INSERT INTO batch_metadata(filename, timestamp, new) VALUES(?, ?, ?)''',
                    (filename, time.time(), 1))
        batch_id = cur.lastrowid
        self.con.commit()
        return batch_id

    def add_row_to_metadata(self, row: int, batch_id: int, score: float):
        """
        Insert a data item from a batch to the row metadata database

        Args:
            row (int): row id to uniquely identify the row
            batch_id (int): id of the corresponding batch
            score (float): initial score of the row
        """
        cur = self.con.cursor()
        cur.execute('''INSERT INTO row_metadata(row, batch_id, score) VALUES(?, ?, ?)''',
                    (row, batch_id, score))
        self.con.commit()

    def update_batch_metadata(self, batch_id: int, score: float, new: bool):
        """
        Update the metadata of a batch

        Args:
            batch_id (int): id of the corresponding batch
            score (float): score to be updated
            new (bool): new flag to be updated
        """
        cur = self.con.cursor()
        cur.execute(
            '''UPDATE batch_metadata SET score = ?, new = ? WHERE id = ?''', (score, new, batch_id))
        self.con.commit()

    def update_row_metadata(self, batch_id: int, row: int, score: float):
        """
        Update the metadata of a data item as a row

        Args:
            batch_id (int): id of the corresponding batch
            row (int): row id to uniquely identify the row
            score (float): score to be updated
        """
        cur = self.con.cursor()
        cur.execute(
            '''UPDATE row_metadata SET score = ? WHERE batch_id = ? AND row = ?''', (score, batch_id, row))
        self.con.commit()

    def get_initial_random_score(self) -> float:
        """
        Generate random initial score

        Returns:
            float: random initial score between 0 and 1
        """
        return random.uniform(0, 1)

    def fetch_batches(self, sql_statment: str, batch_count: int) -> list[tuple[int, str]]:
        """
        Fetch the corresponding batches accoring to the sql statement

        Args:
            sql_statment (str): sql statement to retrieve batches
            batch_count (int): number of batches to be retrieved

        Returns:
            list[tuple[int, str]]: list of tuples of batch ids to filename
        """
        cur = self.con.cursor()
        cur.execute(sql_statment, (batch_count))
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
        cur = self.con.cursor()
        cur.execute(sql_statement, (batch_id, row_count))
        rows = cur.fetchall()
        return rows

    def update_batches(self, batch_selection=BATCHES_BY_SCORE, row_selection=ROWS_BY_SCORE, batch_count=config['data_orchestrator']['nr_batches_update'], batch_size=config['data_feeder']['batch_size']):
        """_summary_

        Args:
            batch_selection (str, optional): sql to select batches according to a criteria. Defaults to BATCHES_BY_SCORE.
            row_selection (str, optional): sql to select rows accordig to a criteria. Defaults to ROWS_BY_SCORE.
            batch_count (int, optional): number of batches to include in the updated batch. Defaults to config['data_orchestrator']['nr_batches_update'].
            batch_size (int, optional): number of rows in the resulting batch. Defaults to config['data_feeder']['batch_size'].
        """
        logging.info('Updating batches')
        batches = self.fetch_batches(batch_selection, batch_count)
        row_count = int(batch_size / batch_count)
        filename_to_rows = dict()
        for batch_id, filename in enumerate(batches):
            rows = self.fetch_rows(row_selection, row_count, batch_id)
            filename_to_rows[filename] = rows

        new_filename, new_rows = self.data_storage.create_shuffled_batch(
            filename_to_rows)
        self.add_batch(new_filename, new_rows)
