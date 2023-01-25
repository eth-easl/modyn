# pylint: disable=singleton-comparison
# flake8: noqa: E712
import numpy as np
from modyn.backend.metadata_database.models.metadata import Metadata
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

    def __init__(self, config: dict, modyn_config: dict):
        super().__init__(config, modyn_config)

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

    def inform_data(self, pipeline_id: int, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        self.database.set_metadata(
            keys,
            timestamps,
            [None] * len(keys),
            [False] * len(keys),
            [None] * len(keys),
            [None] * len(keys),
            pipeline_id,
        )

    def trigger(self) -> None:
        # For data freshness strategy, no prep work to do.
        pass

    def select_new_training_samples(self, training_id: int) -> list[tuple[str, float]]:
        """
        Selects a new training set of samples for the given training id.

        Args:
            training_id (int): The training ID of the current training.
            training_set_size (int): The size of the training set queried.

        Returns:
            list(tuple[str, float]): the training sample keys for the newly selected training_set, along with
                a weight of 1 for each element.
        """
        unseen_data_ratio = self.unseen_data_ratio
        if self._is_adaptive_ratio:
            unseen_data_ratio = self._get_adaptive_unseen_ratio(training_id)

        if self.training_set_size_limit > 0:
            new_samples, old_samples = self._select_new_training_samples_with_limit(training_id, unseen_data_ratio)
        else:
            new_samples, old_samples = self._select_new_training_samples_without_limit(training_id, unseen_data_ratio)

        new_samples.extend(old_samples)
        return [(sample, 1.0) for sample in new_samples]

    def _select_new_training_samples_with_limit(
        self, training_id: int, unseen_data_ratio: float
    ) -> tuple[list[str], list[str]]:
        """If there is a limit, then we should return the proper proportion.


        Args:
            training_id (int): The training ID of the current training

        Returns:
            tuple[list[str], list[str]]: A tuple of new_samples, old_samples
        """
        num_new_samples = int(self.training_set_size_limit * unseen_data_ratio)
        num_old_samples = self.training_set_size_limit - num_new_samples
        new_samples = self._get_unseen_data(training_id, num_new_samples)
        old_samples = self._get_seen_data(training_id, num_old_samples)
        return new_samples, old_samples

    def _select_new_training_samples_without_limit(
        self, training_id: int, unseen_data_ratio: float
    ) -> tuple[list[str], list[str]]:
        """If there is no limit, and we have a strict ratio to maintain (not adaptive), then
        we have to use the relatively smaller amount. For example, let's say we have 25/75
        new/old ratio. If we have 40 new points and 60 old points, we can only return 80
        points (20 new / 60 old).
        If there is no strict ratio, then we can return all the data.

        Args:
            training_id (int): The training ID of the current training
            unseen_data_ratio (float): The desired ratio of unseen data to seen data, if applicable

        Returns:
            tuple[list[str], list[str]]: A tuple of new_samples, old_samples
        """
        if self._is_adaptive_ratio:
            new_samples = self._get_unseen_data(training_id, -1)
            old_samples = self._get_seen_data(training_id, -1)
        else:
            if unseen_data_ratio == 0.0:
                new_samples = []
                old_samples = self._get_seen_data(training_id, -1)
            elif unseen_data_ratio == 1.0:
                new_samples = self._get_unseen_data(training_id, -1)
                old_samples = []
            else:
                num_new_samples = self._get_unseen_data_size(training_id)
                num_old_samples = self._get_seen_data_size(training_id)
                new_samples_multiple = int(num_new_samples / unseen_data_ratio)
                old_samples_multiple = int(num_old_samples / (1 - unseen_data_ratio))
                total_samples = min(new_samples_multiple, old_samples_multiple)
                if new_samples_multiple < old_samples_multiple:
                    new_samples = self._get_unseen_data(training_id, -1)
                    old_samples = self._get_seen_data(training_id, total_samples - num_new_samples)
                else:
                    new_samples = self._get_unseen_data(training_id, total_samples - num_old_samples)
                    old_samples = self._get_seen_data(training_id, -1)
        return new_samples, old_samples

    def _get_unseen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many previously unseen samples.

        Args:
            training_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried. If negative, returns all.

        Returns:
            List of keys for the unseen samples.
        """
        data = (
            self.database.session.query(Metadata.key, Metadata.seen)
            .filter(
                Metadata.training_id == training_id,
                Metadata.seen == False,
            )
            .all()
        )
        if len(data) > 0:
            keys, seen = zip(*data)
        else:
            keys, seen = [], []

        assert len(seen) == 0 or not np.array(seen).any(), "Queried unseen data, but got seen data."
        if num_samples < 0 or num_samples > len(keys):
            num_samples = len(keys)
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        result = list(np.array(keys)[choice])
        if num_samples < 0:
            assert len(result) == num_samples
        return result

    def _get_seen_data(self, training_id: int, num_samples: int) -> list[str]:
        """
        For a given training_id and number of samples, request that many samples from
        the previously seen data

        Args:
            training_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried. If negative, returns all.

        Returns:
            List of keys for the previously seen samples
        """
        data = (
            self.database.session.query(Metadata.key, Metadata.seen)
            .filter(Metadata.training_id == training_id, Metadata.seen == True)
            .all()
        )
        if len(data) > 0:
            keys, seen = zip(*data)
        else:
            keys, seen = [], []

        assert len(seen) == 0 or np.array(seen).all(), "Queried seen data, but got unseen data."
        if num_samples < 0 or num_samples > len(keys):
            num_samples = len(keys)
        choice = np.random.choice(len(keys), size=num_samples, replace=False)
        result = list(np.array(keys)[choice])
        if num_samples < 0:
            assert len(result) == num_samples
        return result

    def _get_seen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many unseen samples there are

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of unseen samples
        """
        data = (
            self.database.session.query(Metadata.key, Metadata.seen)
            .filter(Metadata.training_id == training_id, Metadata.seen == True)
            .all()
        )
        assert len(data) > 0, "Queried unseen data, but got seen data."
        keys, seen = zip(*data)

        assert np.array(seen).all(), "Queried seen data, but got unseen data."
        return len(keys)

    def _get_unseen_data_size(self, training_id: int) -> int:
        """For a given training_id, return how many previously seen samples there are

        Args:
            training_id (int): the queried training_id

        Returns:
            int: number of previously seen samples
        """
        data = (
            self.database.session.query(Metadata.key, Metadata.seen)
            .filter(Metadata.training_id == training_id, Metadata.seen == False)
            .all()
        )
        assert len(data) > 0, "Queried unseen data, but got seen data."
        keys, seen = zip(*data)

        assert not np.array(seen).any(), "Queried unseen data, but got seen data."
        return len(keys)
