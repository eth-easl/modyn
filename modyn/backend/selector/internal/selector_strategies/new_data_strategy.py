# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from typing import Iterable

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from modyn.utils import window_query
from sqlalchemy import asc

logger = logging.getLogger(__name__)


class NewDataStrategy(AbstractSelectionStrategy):
    """
    This strategy always uses the latest data to train on.

    If we reset after trigger, we always use the data since the last trigger.
    If there is a limit, we choose a random subset from that data.
    This configuration can be used to either continously finetune
    or retrain from scratch on only new data.

    Without reset, we always output all data points since the pipeline has been started.
    If we do not have a limit, this can be used to retrain from scratch on all data.
    Finetuning does not really make sense here, unless you want to revisit all data all the time.
    If we have a limit, we use the "limit_reset" configuration option in the config dict to set a strategy.
    Currently we support "lastX" to train on the last LIMIT datapoints (ignoring triggers because we do not reset).
    We also support "sampleUAR" to sample a subset uniformly at random out of all data points.
    TODO(#125): In the future, we might want to support a sampling strategy that prioritizes newer data in some way.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int):
        super().__init__(config, modyn_config, pipeline_id)

        if self.has_limit and not self.reset_after_trigger and "limit_reset" not in config:
            raise ValueError("Please define how to deal with the limit without resets using the 'limit_reset' option.")

        if not (self.has_limit and not self.reset_after_trigger) and "limit_reset" in config:
            logger.warning("Since we do not have a limit and not reset, we ignore the 'limit_reset' setting.")

        self.supported_limit_reset_strategies = ["lastX", "sampleUAR"]
        if "limit_reset" in config:
            self.limit_reset_strategy = config["limit_reset"]

            if self.limit_reset_strategy not in self.supported_limit_reset_strategies:
                raise ValueError(f"Unsupported limit reset strategy: {self.limit_reset_strategy}")

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        self._persist_samples(keys, timestamps, labels)

    def _on_trigger(self) -> Iterable[list[tuple[str, float]]]:
        """
        Internal function. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

        TODO(MaxiBoether): update docstring

        Returns:
            list(tuple(str, float)): Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        if self.reset_after_trigger:
            get_data_func = self._get_data_reset
        else:
            get_data_func = self._get_data_no_reset

        for samples in get_data_func():
            random.shuffle(samples)
            yield [(sample, 1.0) for sample in samples]

    def _get_data_reset(self) -> Iterable[list[str]]:
        assert self.reset_after_trigger

        for samples in self._get_current_trigger_data():
            # TODO(create issue): this assumes limit < len(samples)
            if self.has_limit:
                samples = random.sample(samples, min(len(samples), self.training_set_size_limit))

            yield samples

    def _get_data_no_reset(self) -> Iterable[list[str]]:
        assert not self.reset_after_trigger
        for samples in self._get_all_data():
            # TODO(create issue): this assumes limit < len(samples)
            if self.has_limit:
                samples = self._handle_limit_no_reset(samples)

            yield samples

    def _handle_limit_no_reset(self, samples: list[str]) -> list[str]:
        assert self.limit_reset_strategy is not None

        if self.limit_reset_strategy == "lastX":
            return self._last_x_limit(samples)

        if self.limit_reset_strategy == "sampleUAR":
            return self._sample_uar(samples)

        raise NotImplementedError(f"Unsupport limit reset strategy: {self.limit_reset_strategy}")

    def _last_x_limit(self, samples: list[str]) -> list[str]:
        assert self.has_limit
        assert self.training_set_size_limit > 0

        return samples[-self.training_set_size_limit :]

    def _sample_uar(self, samples: list[str]) -> list[str]:
        assert self.has_limit
        assert self.training_set_size_limit > 0

        return random.sample(samples, min(len(samples), self.training_set_size_limit))

    def _get_current_trigger_data(self) -> Iterable[list[str]]:
        """Returns all sample for current trigger

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            query = (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id == self._next_trigger_id,
                )
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            for chunk in window_query(query, SelectorStateMetadata.timestamp, self._maximum_keys_in_memory, False):
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _get_all_data(self) -> Iterable[list[str]]:
        """Returns all sample

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            query = (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id)
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            for chunk in window_query(query, SelectorStateMetadata.timestamp, self._maximum_keys_in_memory, False):
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.
