from abc import ABC, abstractmethod

from .. import Metadata


class Scorer(Metadata):
    """
    Abstract class to score the samples according to the implementors defined data importance measure
    and update the corresponding metadata database
    """
    _con = None
    _nr_training_setes = 0
    _training_set_selection = None
    _sample_selection = None

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information

            data_storage (DataStorage): Data storage instance to store the newly created data instances
        """
        super().__init__(config)

    def _add_training_set(
            self,
            filename: str,
            samples: list[int],
            initial=True,
            training_set_id=None) -> int:
        """
        Add a training_set to the data scorer metadata database

        Calculate the training_set score by the score defined by the subclass of the sample scores

        Args:
            filename (str): filename of the training_set
            samples (list[int]): sample numbers in the training_set
            initial (bool): if adding a training_set initially or otherwise

        Returns:
            int: training_set_id of the added training_set
        """
        if (initial and self._nr_training_setes %
                self._config['metadata']['nr_files_update'] == 0):
            self._create_shuffled_training_setes(
                self._config['metadata']['nr_files_update'],
                self._config['feeder']['training_set_size'])
        if training_set_id is None:
            training_set_id = self._add_training_set_to_metadata()

        scores = []
        for sample in samples:
            score = self._get_score()
            self._add_sample_to_metadata(
                sample, training_set_id, score, filename)
            scores.append(score)
        if initial:
            training_set_score = self._get_cumulative_score(scores)
            self._nr_training_setes += 1
            self._update_training_set_metadata(
                training_set_id, training_set_score, 1)
        return training_set_id

    def _create_shuffled_training_setes(
            self, training_set_count: int, training_set_size: int):
        """
        Update an existing training_set based on a training_set and sample selection criterion. Select a number
        of training_sets and decide on a total number of samples for the new training_set. The created
        training_set will contain equal proportion of the selected training_setes.

        Args:
            training_set_count (int, optional): number of training_setes to include in the updated training_set.
            training_set_size (int, optional): number of samples in the resulting training_set.
        """
        training_setes = self._fetch_training_set_ids(
            self._training_set_selection, training_set_count)
        sample_count = int(training_set_size / training_set_count)
        self._nr_training_setes += 1

        new_training_set_id = None

        for training_set_id in training_setes:
            samples = self._fetch_filenames_to_indexes(
                self._sample_selection, sample_count, training_set_id)

            for filename in samples:
                bid = self._add_training_set(
                    filename,
                    samples[filename],
                    initial=False,
                    training_set_id=new_training_set_id)

                if new_training_set_id is None:
                    new_training_set_id = bid
        if new_training_set_id is not None:
            new_training_set_scores = self._get_scores(new_training_set_id)
            new_training_set_score = self._get_cumulative_score(
                new_training_set_scores)
            self._update_training_set_metadata(
                new_training_set_id, new_training_set_score, 1)

    def add_training_set(
            self, filename: str, samples: list[int]):
        """
        Add a training_set to the data scorer metadata database

        Args:
            filename (str): filename of the training_set
            samples (list[int]): sample numbers in the training_set

        Returns:
            int: training_set_id of the added training_set
        """
        return self._add_training_set(filename, samples)

    @abstractmethod
    def _get_score(self) -> float:
        """
        Generate score according to data importance

        Returns:
            float: score
        """
        raise NotImplementedError

    @abstractmethod
    def _get_cumulative_score(self, scores: list[int]):
        """
        Calculate the cumulative score of a list of scores

        Args:
            scores (list[int]): list of scores

        Returns:
            float: cumulative score
        """
        raise NotImplementedError
