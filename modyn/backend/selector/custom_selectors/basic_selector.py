
from modyn.backend.selector.selector import Selector
import numpy as np


class BasicSelector(Selector):
    """
    This class implements selection solely based on freshness of the data.
    Specifically, there is a "unseen_data_ratio" that controls
    how much of each batch is from unseen data, and how much is from previously
    seen data. If is_adaptive_ratio is set to True, then this ratio is automatically
    set to the proportion of the size of the unseen vs. previously seen data.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.unseen_data_ratio = 1.0
        self.old_data_ratio = 0.0
        self._set_unseen_data_ratio(self._config['selector']['unseen_data_ratio'])
        self._set_is_adaptive_ratio(self._config['selector']['is_adaptive_ratio'])

    def _set_unseen_data_ratio(self, unseen_data_ratio: float) -> None:
        assert 0 <= unseen_data_ratio <= 1
        self.unseen_data_ratio = unseen_data_ratio
        self.old_data_ratio = 1 - self.unseen_data_ratio

    def _set_is_adaptive_ratio(self, is_adaptive_ratio: bool) -> None:
        self._is_adaptive_ratio = is_adaptive_ratio

    def _select_new_training_samples(
            self,
            training_id: int,
            training_set_size: int
    ) -> list[tuple[str]]:
        """
        Selects a new training set of samples for the given training id.

        Returns:
            list(str): the training sample keys for the newly selected training_set
        """
        if self._is_adaptive_ratio:
            seen_data_size = self._get_seen_data_size(training_id)
            unseen_data_size = self._get_unseen_data_size(training_id)
            self.unseen_data_ratio = unseen_data_size / (unseen_data_size + seen_data_size)

        num_new_samples = int(training_set_size * self.unseen_data_ratio)
        num_old_samples = training_set_size - num_new_samples
        new_samples = self._get_unseen_data(training_id, num_new_samples)
        old_samples = self._get_seen_data(training_id, num_old_samples)
        new_samples.extend(old_samples)
        return [(sample,) for sample in new_samples]

    def _get_unseen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many previously unseen samples.

        Returns:
            List of keys for the unseen samples.
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 0 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self.get_samples_by_metadata_query(query)
        assert len(seen) == 0 or not np.array(seen).any(), "Queried unseen data, but got seen data."
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return np.array(keys)[choice]

    def _get_seen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the previously seen data

        Returns:
            List of keys for the previously seen samples
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 1 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self.get_samples_by_metadata_query(query)
        assert len(seen) == 0 or np.array(seen).all(), "Queried seen data, but got unseen data."
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return np.array(keys)[choice]

    def _get_seen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many unseen samples there are

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of unseen samples
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 1 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self.get_samples_by_metadata_query(query)
        assert np.array(seen).all(), "Queried seen data, but got unseen data."
        return len(keys)

    def _get_unseen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many previously seen samples there are

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of previously seen samples
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 0 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self.get_samples_by_metadata_query(query)
        assert not np.array(seen).any(), "Queried unseen data, but got seen data."
        return len(keys)
