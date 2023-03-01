import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata, Trigger, TriggerSample
from sqlalchemy import func

from modyn.utils import window_query

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
        self._maximum_keys_in_memory = 500000 # TODO(MaxiBoether): add config option

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
    def _on_trigger(self) -> Iterable[list[tuple[str, float]]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

        TODO(MaxiBoether): update below

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

    def trigger(self) -> tuple[int, int]:
        """
        Causes the strategy to compute the training set, and (if so configured) reset its internal state.

        Returns:
            tuple[int, int]: Trigger ID and number of keys that define the trigger
        """
        trigger_id = self._next_trigger_id

        total_keys_in_trigger = 0

        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.session.add(Trigger(pipeline_id=self._pipeline_id, trigger_id=trigger_id))
            database.session.commit()   

            for training_samples in self._on_trigger():
                logger.info(
                    "Strategy for pipeline {} returned batch of {} samples for new trigger {}.",
                    self._pipeline_id,
                    len(training_samples),
                    trigger_id,
                )

                total_keys_in_trigger += len(training_samples)

                database.session.bulk_save_objects(
                    [
                        TriggerSample(trigger_id=trigger_id, pipeline_id=self._pipeline_id, sample_key=key)
                        for key, _ in training_samples
                    ]
                )
            
            database.session.commit()

        if self.reset_after_trigger:
            self._reset_state()

        self._next_trigger_id += 1
        return trigger_id, total_keys_in_trigger
    
    def get_trigger_keys(self, trigger_id: int) -> Iterable[list[tuple[str, float]]]:
        # TODO(MaxiBoether): Write docstring

        # TODO(MaxiBoether): CHANGE THIS TO get_trigger_partition_keys
        # Instead of generator, what we want is to work with indices of partitions.
        # We _know_ how many partitions there are for this trigger (we should persist it to DB as well)
        # Then, we can call get_trigger_keys(trigger_id, partition_id) with partition_id in [0 ... num_partitions - 1]
        # We only load the specific window into memory when this is called
        # For the workers, we divide [0 ... num_partitions - 1] into equally sized ranges, i.e., each worker has the same number of partitions
        # Each worker can ask how many partitions there are for this worker (should implement at selector!). Be careful about the ende dass wir nicht zu viele Partitions returnen
        # Here, at get_trigger_keys, we just care about the global trigger ID
        # At the selector, we have get_sample_keys_and_weights(trigger_id, worker_id, partition_id)
        # We have to convert the worker partition id into the global partition id
        # With this global partition id, we then either check the cache and get the global partition of that data
        # Or we ask get_trigger_keys, if not cached, about that partition

        # After this is done, we need to change the strategies as well to implement _on_trigger as a generator
        # And never materialize everything.

        with MetadataDatabaseConnection(self._modyn_config) as database:
            query = (
                database.session.query(TriggerSample.sample_key, TriggerSample.sample_weight)
                .filter(TriggerSample.pipeline_id == self._pipeline_id, TriggerSample.trigger_id == trigger_id)
            )

            for chunk in window_query(query, "trigger_sample_list_id", self._maximum_keys_in_memory):
                yield [(row[0], row[1]) for row in chunk]


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
