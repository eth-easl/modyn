import random
import os
import statistics

from . import Scorer

STORAGE_LOCATION = os.getcwd()


class RandomScorer(Scorer):
    """
    Score the samples according to the defined data importance measure and update the corresponding metadata database
    """
    _sample_selection = '''SELECT filename, sample
                      FROM sample_metadata
                      JOIN training_set_to_sample ON training_set_to_sample.sample = sample_metadata.sample
                      WHERE training_set_id=%s
                      ORDER BY score DESC LIMIT '''
    _training_set_selection = '''SELECT id FROM training_set_metadata ORDER BY score DESC LIMIT '''

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
