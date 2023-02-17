import logging
from abc import ABC, abstractmethod
from typing import Optional

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata, Trigger, TriggerSample
from sqlalchemy import func

logger = logging.getLogger(__name__)


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selection strategies.
    New selection strategies need to implement the
    `_on_trigger`, `_reset_state`, and `inform_data` methods.

    Args:
        config (dict): the configurations for the selector
        modyn_config (dict): the configurations for the modyn backend
    """

    def __init__(
        self, config: dict, modyn_config: dict, pipeline_id: int, required_configs: Optional[list[str]] = None
    ):
        self._config = config

        if required_configs is None:
            required_configs = []  # Using [] as default is considered unsafe by pylint

        required_configs.extend(["limit", "reset_after_trigger"])
        for required_config in required_configs:
            if required_config not in self._config.keys():
                raise ValueError(f"{required_config} not given but required.")

        self.training_set_size_limit: int = config["limit"]
        self.has_limit = self.training_set_size_limit > 0
        self.reset_after_trigger: bool = config["reset_after_trigger"]
        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id

        logger.info(f"Initializing selection strategy for pipeline {pipeline_id}.")

        with MetadataDatabaseConnection(self._modyn_config) as database:
            last_trigger_id = (
                database.session.query(func.max(Trigger.trigger_id))  # pylint: disable=not-callable
                .filter(Trigger.pipeline_id == self._pipeline_id)
                .scalar()
            )
            if last_trigger_id is None:
                logger.info(f"Did not find previous trigger id DB for pipeline {pipeline_id}, next trigger is 0.")
                self._next_trigger_id = 0
            else:
                logger.info(f"Last trigger in DB for pipeline {pipeline_id} was {last_trigger_id}.")
                self._next_trigger_id = last_trigger_id + 1

    @abstractmethod
    def _on_trigger(self) -> list[tuple[str, float]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on.

        Returns:
            list(tuple(str, float)): Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_state(self) -> None:
        """Resets the internal state of the strategy, e.g., by clearing buffers."""
        raise NotImplementedError

    @abstractmethod
    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        """Informs the strategy of new data.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
        """
        raise NotImplementedError

    def trigger(self) -> tuple[int, list[tuple[str, float]]]:
        """
        Causes the strategy to compute the training set, and (if so configured) reset its internal state.

        Returns:
            tuple[int, list[tuple[str, float]]]: Trigger ID and a list of the training data.
              In this list, each entry is a training sample,
              where the first element of the tuple is the key, and the second element is the associated weight.
        """
        trigger_id = self._next_trigger_id
        training_samples = self._on_trigger()

        logger.info(
            f"Strategy for pipeline {self._pipeline_id} got"
            + "f{len(training_samples)} samples for new trigger {trigger_id}."
        )

        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.session.add(Trigger(pipeline_id=self._pipeline_id, trigger_id=trigger_id))
            database.session.commit()
            database.session.add_all(
                [
                    TriggerSample(trigger_id=trigger_id, pipeline_id=self._pipeline_id, sample_key=key)
                    for key, _ in training_samples
                ]
            )
            database.session.commit()

        if self.reset_after_trigger:
            self._reset_state()

        self._next_trigger_id += 1
        return trigger_id, training_samples

    def _persist_samples(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        """Persists the data in the database.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
            labels (list[int]): A list of labels of the data.
            database (MetadataDatabaseConnection): The database connection.
        """
        # TODO(#116): Right now we persist all datapoint into DB. We might want to
        # keep this partly in memory for performance.
        # Even if each sample is 64 byte and we see 2 million samples, it's just 128 MB of data in memory.
        # This also means that we have to clear this list on reset accordingly etc.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            new_selector_state_metadata = [
                SelectorStateMetadata(
                    pipeline_id=self._pipeline_id,
                    sample_key=key,
                    timestamp=timestamp,
                    label=label,
                    seen_in_trigger_id=self._next_trigger_id,
                )
                for key, timestamp, label in zip(keys, timestamps, labels)
            ]
            database.session.add_all(new_selector_state_metadata)
            database.session.commit()
