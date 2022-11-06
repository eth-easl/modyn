import sqlite3
import random
import logging
import os
import statistics
import uuid

from . import Scorer

STORAGE_LOCATION = os.getcwd()


class RandomScorer(Scorer):
    """
    Score the samples according to the defined data importance measure and update the corresponding metadata database
    """
    _config = None
    _con = None
    __nr_batches_update = 3
    __batch_size = 100
    _row_selection = "SELECT filename, row from row_metadata WHERE batch_id=? ORDER BY score DESC LIMIT "
    _batch_selection = "SELECT id FROM batch_metadata ORDER BY score DESC LIMIT "

    def __init__(self, config: dict):
        """
        Args:
            __config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        self.__config = config
        self.__nr_batches_update = config['data_scorer']['nr_files_update']
        self.__batch_size = config['data_feeder']['batch_size']
        self._con = sqlite3.connect(
            config['data_scorer']['in_memory_database'])
        self._setup_database()

    def _get_score(self) -> float:
        """
        Generate random initial score

        Returns:
            float: random initial score between 0 and 1
        """
        return random.uniform(0, 1)

    def _get_cumulative_score(self, scores: list[int]):
        """
        Calculate the cumulative score of a list of scores

        Args:
            scores (list[int]): list of scores

        Returns:
            float: cumulative score
        """
        return statistics.median(scores)
