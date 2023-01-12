import numpy as np
from modyn.backend.selector.internal.grpc.grpc_handler import GRPCHandler
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class DataFreshnessStrategy(AbstractSelectionStrategy):
    """
    This class implements selection solely based on freshness of the data.
    Specifically, there is a "unseen_data_ratio" that controls
    how much of each batch is from unseen data, and how much is from previously
    seen data. If is_adaptive_ratio is set to True, then this ratio is automatically
    set to the proportion of the size of the unseen vs. previously seen data.

    For example, if is_adaptive_ratio is set to True, and there are 600 previously
    seen data points and 200 previously unseen data points, we will set unseen_data_ratio
    to 0.25, since there are 200 unseen points and 800 total points, so 200/800=0.25

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, grpc: GRPCHandler):
        super().__init__(config, grpc)

        self.unseen_data_ratio = 1.0
        self.old_data_ratio = 0.0
        self._set_unseen_data_ratio(self._config["selector"]["unseen_data_ratio"])
        self._is_adaptive_ratio = self._config["selector"]["is_adaptive_ratio"]

    def _set_unseen_data_ratio(self, unseen_data_ratio: float) -> None:
        assert 0 <= unseen_data_ratio <= 1
        self.unseen_data_ratio = unseen_data_ratio
        self.old_data_ratio = 1 - self.unseen_data_ratio

    def _get_adaptive_unseen_ratio(self, training_id: int) -> float:
        """Returns the proper adaptive unseen data ratio. For example, if there are 200
        unseen data points and 600 previously unseen data points, it will return
        a ratio of 200 (unseen) / 800 (total) = 0.25

        Args:
            training_id (int): The training ID of the current training.

        Returns:
            float: The unseen ratio.
        """
        seen_data_size = self._get_seen_data_size(training_id)
        unseen_data_size = self._get_unseen_data_size(training_id)
        unseen_data_ratio = unseen_data_size / (unseen_data_size + seen_data_size)
        return unseen_data_ratio

    def _select_new_training_samples(self, training_id: int, training_set_size: int) -> list[tuple[str]]:
        """
        Selects a new training set of samples for the given training id.

        Args:
            training_id (int): The training ID of the current training.
            training_set_size (int): The size of the training set queried.

        Returns:
            list(str): the training sample keys for the newly selected training_set
        """
        unseen_data_ratio = self.unseen_data_ratio
        if self._is_adaptive_ratio:
            unseen_data_ratio = self._get_adaptive_unseen_ratio(training_id)

        num_new_samples = int(training_set_size * unseen_data_ratio)
        num_old_samples = training_set_size - num_new_samples
        new_samples = self._get_unseen_data(training_id, num_new_samples)
        old_samples = self._get_seen_data(training_id, num_old_samples)
        new_samples.extend(old_samples)
        return [(sample,) for sample in new_samples]

    def _get_unseen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many previously unseen samples.

        Args:
            training_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried

        Returns:
            List of keys for the unseen samples.
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 0 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self._grpc.get_samples_by_metadata_query(query)
        assert len(seen) == 0 or not np.array(seen).any(), "Queried unseen data, but got seen data."
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return list(np.array(keys)[choice])

    def _get_seen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the previously seen data

        Args:
            training_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried

        Returns:
            List of keys for the previously seen samples
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 1 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self._grpc.get_samples_by_metadata_query(query)
        assert len(seen) == 0 or np.array(seen).all(), "Queried seen data, but got unseen data."
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        return list(np.array(keys)[choice])

    def _get_seen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many unseen samples there are

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of unseen samples
        """
        query = f"""SELECT key, score, seen, label, data FROM metadata_database
                 WHERE seen = 1 AND training_id = {training_id}"""
        keys, _, seen, _, _ = self._grpc.get_samples_by_metadata_query(query)
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
        keys, _, seen, _, _ = self._grpc.get_samples_by_metadata_query(query)
        assert not np.array(seen).any(), "Queried unseen data, but got seen data."
        return len(keys)
