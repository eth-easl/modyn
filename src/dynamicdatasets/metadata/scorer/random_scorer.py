import sqlite3
import random
import logging
import os
import statistics

from . import Scorer

STORAGE_LOCATION = os.getcwd()


class RandomScorer(Scorer):
    """
    Score the samples according to the defined data importance measure and update the corresponding metadata database
    """
    _config = None
    _data_storage = None
    _con = None
    __nr_batches_update = 3
    __batch_size = 100
    ROWS_BY_SCORE = "SELECT row, score from row_metadata WHERE batch_id=? ORDER BY score DESC LIMIT "
    BATCHES_BY_SCORE = "SELECT id, filename FROM batch_metadata ORDER BY score DESC LIMIT "
    BATCHES_BY_TIMESTAMP = "SELECT id, filename FROM batch_metadata ORDER BY timestamp DESC LIMIT "

    def __init__(self, config: dict):
        """
        Args:
            __config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        self.__config = config
        self._data_storage = None  # TODO: Replace with remote procedure call
        self.__nr_batches_update = config['data_scorer']['nr_files_update']
        self.__batch_size = config['data_feeder']['batch_size']
        self._con = sqlite3.connect(
            config['data_scorer']['in_memory_database'])
        self.setup_database()

    def get_score(self) -> float:
        """
        Generate random initial score

        Returns:
            float: random initial score between 0 and 1
        """
        return random.uniform(0, 1)

    def create_shuffled_batches(
            self, batch_selection, row_selection, batch_count, batch_size):
        """
        Update an existing batch based on a batch and row selection criterion. Select a number of batches
        and decide on a total number of samples for the new batch. The created batch will contain equal
        proportion of the selected batches.

        Args:
            batch_selection (str, optional): sql to select batches according to a criteria.
            row_selection (str, optional): sql to select rows accordig to a criteria.
            batch_count (int, optional): number of batches to include in the updated batch.
            batch_size (int, optional): number of rows in the resulting batch.
        """
        batches = self.fetch_batches(batch_selection, batch_count)
        row_count = int(batch_size / batch_count)
        filename_to_rows = dict()
        for batch_id, filename in batches:
            rows = self.fetch_rows(row_selection, row_count, batch_id)
            rows.sort()
            filename_to_rows[filename] = rows

        new_filename, new_rows = self._data_storage.create_shuffled_batch(
            filename_to_rows)
        self.add_batch(new_filename, new_rows, initial=False)

    def add_batch(self, filename: str, rows: list[int], initial=True):
        """
        Add a batch to the data scorer metadata database

        Calculate the batch score by the median of the row scores

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch
            initial (bool): if adding a batch initially or otherwise
        """
        logging.info(
            f'Adding batch from input file {filename} to metadata database')
        self._nr_batches += 1
        if (initial and self._nr_batches % self._config['data_scorer']['nr_files_update'] == 0):
            self.create_shuffled_batches(
                self.BATCHES_BY_SCORE, self.ROWS_BY_SCORE,
                self._config['data_scorer']['nr_files_update'],
                self._config['data_feeder']['batch_size'])
        batch_id = self.add_batch_to_metadata(filename)
        scores = []
        for row in rows:
            score = self.get_score()
            self.add_row_to_metadata(row, batch_id, score)
            scores.append(score)
        median = statistics.median(scores)
        self.update_batch_metadata(batch_id, median, 1)
