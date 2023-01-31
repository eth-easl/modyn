# TODO(MaxiBoether): implement.
# Idea: With reset, we always spit out the data since last trigger.
# If we have a limit, we choose a random subset from the data.
# This can be used to either continously finetune or retrain from scratch on only new data.
# Without reset, we always spit out the entire new data.
# If we do not have a limit, this can be used to retrain from scratch on all data,
# finetuning does not really make sense here, unless you want to revisit all the time.
# If we have a limit and no reset, we have multiple options and need to decide on one: "last X samples",
# sample uniform at random from all data, sample from all data but prioritize newer data points in some way

# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
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
    TODO(create issue): In the future, we might want to support a sampling strategy that prioritizes newer data in some way.

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

        # TODO(#116): Right now we persist all datapoint into DB. We might want to keep this partly in memory for performance.
        # Even if each sample is 64 byte and we see 2 million samples, it's just 128 MB of data in memory.
        # This also means that we have to clear this list on reset accordingly etc.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.set_metadata(
                keys,
                timestamps,
                [None] * len(keys),
                [False] * len(keys),
                labels,
                [None] * len(keys),
                self._pipeline_id,
                self._next_trigger_id,
            )

    def _on_trigger(self) -> list[tuple[str, float]]:
        """
        Internal function. Calculates the next set of data to
        train on.

        Returns:
            list(tuple(str, float)): Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """

        if self.reset_after_trigger:
            samples = self._get_data_reset()
        else:
            samples = self._get_data_no_reset()

        random.shuffle(samples)

        return [(sample, 1.0) for sample in samples]

    def _get_data_reset(self) -> list[str]:
        assert self.reset_after_trigger
        samples = self._get_current_trigger_data()
        if self.has_limit:
            samples = random.sample(samples, min(len(samples), self.training_set_size_limit))

        return samples

    def _get_data_no_reset(self) -> list[str]:
        assert not self.reset_after_trigger
        samples = self._get_all_data()

        if self.has_limit:
            samples = self._handle_limit_no_reset(samples)

        return samples

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

    def _get_current_trigger_data(self) -> list[str]:
        """Returns all sample for current trigger

        Returns:
            list[str]: Keys of used samples
        """
        logger.error(f"{self._next_trigger_id} - lol")
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.pipeline_id == self._pipeline_id, Metadata.trigger_id == self._next_trigger_id)
                .order_by(asc(Metadata.timestamp))
                .all()
            )

        if len(data) > 0:
            keys, _ = zip(*data)
        else:
            keys = []

        return list(keys)

    def _get_all_data(self) -> list[str]:
        """Returns all sample

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.pipeline_id == self._pipeline_id)
                .order_by(asc(Metadata.timestamp))
                .all()
            )

        if len(data) > 0:
            keys, _ = zip(*data)
        else:
            keys = []

        return list(keys)

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.
