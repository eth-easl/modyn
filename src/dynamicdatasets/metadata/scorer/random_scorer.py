import random
import os
import statistics

from . import Scorer

STORAGE_LOCATION = os.getcwd()


class RandomScorer(Scorer):
    """
    Score the samples according to the defined data importance measure and update the corresponding metadata database
    """
    _row_selection = '''SELECT filename, row
                      FROM row_metadata
                      JOIN batch_to_row ON batch_to_row.row = row_metadata.row
                      WHERE batch_id=%s
                      ORDER BY score DESC LIMIT '''
    _batch_selection = '''SELECT id FROM batch_metadata ORDER BY score DESC LIMIT '''

    def __init__(self, config: dict):
        """
        Args:
            __config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        super().__init__(config)

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
