# pylint: disable=singleton-comparison
# flake8: noqa: E712
import numpy as np
from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy


class FreshnessSamplingStrategy(AbstractSelectionStrategy):
    """
    This class selects data from a mixture of used and unsed data.
    We can set a ratio that defines how much data in the training set per trigger should be from previously unused data (in all previous triggers).

    The first trigger will always use only fresh data (up to the limit, if there is one).
    The subsequent triggers will sample a dataset that reflects the ratio of used/unused data (if data came during but was not used in a previous trigger, we still handle it as unseen).
    We have to respect both the ratio and the limit (if there is one) and build up the dataset on trigger accordingly.

    It cannot be used with reset, because we need to keep state over multiple triggers.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int):
        super().__init__(config, modyn_config, pipeline_id, required_configs=["unused_data_ratio"])
        self.unused_data_ratio = self._config["unused_data_ratio"]

        if self.unused_data_ratio < 1 or self.unused_data_ratio > 99:
            raise ValueError(
                f"Invalid unused data ratio: {self.unused_data_ratio}. We need at least 1% fresh data (otherwise we would always train on the data from first trigger) and at maximum 99% fresh data (otherwise please use NewDataStrategy)."
            )

        if self.reset_after_trigger:
            raise ValueError(
                "FreshnessSamplingStrategy cannot reset state after trigger, because then no old data would be available to sample from."
            )

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        # TODO(#116): Right now we persist all datapoint into DB. We might want to keep this partly in memory for performance.

        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.set_metadata(
                keys,
                timestamps,
                [None] * len(keys),
                [False] * len(keys),
                [None] * len(keys),
                [None] * len(keys),
                self._pipeline_id,
                self._next_trigger_id,
            )

    def _on_trigger(self, pipeline_id: int) -> list[tuple[str, float]]:
        """
        Selects a new training set of samples for the given training id.

        Args:
            pipeline_id (int): The training ID of the current training.
            training_set_size (int): The size of the training set queried.

        Returns:
            list(tuple[str, float]): the training sample keys for the newly selected training_set, along with
                a weight of 1 for each element.
        """
        unseen_data_ratio = self.unseen_data_ratio
        if self._is_adaptive_ratio:
            unseen_data_ratio = self._get_adaptive_unseen_ratio(pipeline_id)

        if self.training_set_size_limit > 0:
            new_samples, old_samples = self._select_new_training_samples_with_limit(pipeline_id, unseen_data_ratio)
        else:
            new_samples, old_samples = self._select_new_training_samples_without_limit(pipeline_id, unseen_data_ratio)

        new_samples.extend(old_samples)
        return [(sample, 1.0) for sample in new_samples]

    def _select_new_training_samples_with_limit(
        self, pipeline_id: int, unseen_data_ratio: float
    ) -> tuple[list[str], list[str]]:
        """If there is a limit, then we should return the proper proportion.


        Args:
            pipeline_id (int): The training ID of the current training

        Returns:
            tuple[list[str], list[str]]: A tuple of new_samples, old_samples
        """
        num_new_samples = int(self.training_set_size_limit * unseen_data_ratio)
        num_old_samples = self.training_set_size_limit - num_new_samples
        new_samples = self._get_unseen_data(pipeline_id, num_new_samples)
        old_samples = self._get_seen_data(pipeline_id, num_old_samples)
        return new_samples, old_samples

    def _select_new_training_samples_without_limit(
        self, pipeline_id: int, unseen_data_ratio: float
    ) -> tuple[list[str], list[str]]:
        """If there is no limit, and we have a strict ratio to maintain (not adaptive), then
        we have to use the relatively smaller amount. For example, let's say we have 25/75
        new/old ratio. If we have 40 new points and 60 old points, we can only return 80
        points (20 new / 60 old).
        If there is no strict ratio, then we can return all the data.

        Args:
            pipeline_id (int): The training ID of the current training
            unseen_data_ratio (float): The desired ratio of unseen data to seen data, if applicable

        Returns:
            tuple[list[str], list[str]]: A tuple of new_samples, old_samples
        """
        if self._is_adaptive_ratio:
            new_samples = self._get_unseen_data(pipeline_id, -1)
            old_samples = self._get_seen_data(pipeline_id, -1)
        else:
            if unseen_data_ratio == 0.0:
                new_samples = []
                old_samples = self._get_seen_data(pipeline_id, -1)
            elif unseen_data_ratio == 1.0:
                new_samples = self._get_unseen_data(pipeline_id, -1)
                old_samples = []
            else:
                num_new_samples = self._get_unseen_data_size(pipeline_id)
                num_old_samples = self._get_seen_data_size(pipeline_id)
                new_samples_multiple = int(num_new_samples / unseen_data_ratio)
                old_samples_multiple = int(num_old_samples / (1 - unseen_data_ratio))
                total_samples = min(new_samples_multiple, old_samples_multiple)
                if new_samples_multiple < old_samples_multiple:
                    new_samples = self._get_unseen_data(pipeline_id, -1)
                    old_samples = self._get_seen_data(pipeline_id, total_samples - num_new_samples)
                else:
                    new_samples = self._get_unseen_data(pipeline_id, total_samples - num_old_samples)
                    old_samples = self._get_seen_data(pipeline_id, -1)
        return new_samples, old_samples

    def _get_unseen_data(self, pipeline_id: int, num_samples: int) -> list[str]:
        """
        For a given pipeline_id and number of samples, request that many previously unseen samples.

        Args:
            pipeline_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried. If negative, returns all.

        Returns:
            List of keys for the unseen samples.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(
                    Metadata.training_id == pipeline_id,
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

    def _get_seen_data(self, pipeline_id: int, num_samples: int) -> list[str]:
        """
        For a given pipeline_id and number of samples, request that many samples from
        the previously seen data

        Args:
            pipeline_id (int): The training ID of the current training.
            num_samples (int): Number of samples queried. If negative, returns all.

        Returns:
            List of keys for the previously seen samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.training_id == pipeline_id, Metadata.seen == True)
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

    def _get_seen_data_size(self, pipeline_id: int) -> int:
        """For a given pipeline_id, return how many unseen samples there are

        Args:
            pipeline_id (int): the queried pipeline_id

        Returns:
            int: number of unseen samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.training_id == pipeline_id, Metadata.seen == True)
                .all()
            )
        assert len(data) > 0, "Queried unseen data, but got seen data."
        keys, seen = zip(*data)

        assert np.array(seen).all(), "Queried seen data, but got unseen data."
        return len(keys)

    def _get_unseen_data_size(self, pipeline_id: int) -> int:
        """For a given pipeline_id, return how many previously seen samples there are

        Args:
            pipeline_id (int): the queried pipeline_id

        Returns:
            int: number of previously seen samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.training_id == pipeline_id, Metadata.seen == False)
                .all()
            )

        assert len(data) > 0, "Queried unseen data, but got seen data."
        keys, seen = zip(*data)

        assert not np.array(seen).any(), "Queried unseen data, but got seen data."
        return len(keys)
