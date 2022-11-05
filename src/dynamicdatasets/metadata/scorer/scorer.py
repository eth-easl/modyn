import sqlite3
import time
from abc import ABC, abstractmethod

from .. import Metadata


class Scorer(Metadata):
    """
    Abstract class to score the samples according to the implementors defined data importance measure
    and update the corresponding metadata database
    """
    _data_storage = None
    _nr_batches_update = 3
    _batch_size = 100
    _nr_batches = 0

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        super().__init__(config)
        self._data_storage = None  # TODO: Replace with remote procedure call
        self.nr_batches_update = config['data_scorer']['nr_files_update']
        self.batch_size = config['data_feeder']['batch_size']

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

    @abstractmethod
    def add_batch(
            self, filename: str, rows: list[int],
            scores=None, initial=False):
        """
        Add a batch to the data scorer metadata database

        Args:
            filename (str): filename of the batch
            rows (list[int]): row numbers in the batch
        """
        raise NotImplementedError
