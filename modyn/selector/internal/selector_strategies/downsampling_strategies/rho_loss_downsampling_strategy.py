import random
from typing import Any, Iterable

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies import AbstractSelectionStrategy
from modyn.selector.internal.selector_strategies.downsampling_strategies import AbstractDownsamplingStrategy
from modyn.selector.internal.selector_strategies.utils import get_trigger_dataset_size
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database import DatabaseStorageBackend
from sqlalchemy import Select, asc, func, select


class RHOLossDownsamplingStrategy(AbstractDownsamplingStrategy):
    def __init__(self, downsampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(downsampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)
        self.holdout_set_ratio = downsampling_config["holdout_set_ratio"]
        self.remote_downsampling_strategy_name = "RemoteRHOLossDownsampling"

    def inform_next_trigger(self, next_trigger_id: int, selector_storage_backend: AbstractStorageBackend) -> None:
        if not isinstance(selector_storage_backend, DatabaseStorageBackend):
            raise ValueError("RHOLossDownsamplingStrategy requires a DatabaseStorageBackend")

        # The logic to train an IL model will be implemented here
        # Step 1: Fetch or create the rho_pipeline_id from metadata database
        #
        # Will create a new table (pipeline_id, rho_pipeline_id) representing the mapping
        # between pipeline_id and rho_pipeline_id, primary key being pipeline_id.
        rho_pipeline_id = self._get_or_create_rho_pipeline_id()
        # Step 2: Prepare the training data for the IL model, by randomly sampling a predefined ratio of samples
        # from next_trigger_id's data and storing them as a TSS with identifier (rho_pipeline_id, next_trigger_id).
        self._prepare_holdout_set(next_trigger_id, rho_pipeline_id, selector_storage_backend)
        # Step 3: Issue training request to the trainer server, with pipeline_id as rho_pipeline_id and trigger_id
        # as next_trigger_id. Wait for the training to complete. Store the model. Record model id in
        # downsampling_params, so that it can be fetched and used for downsampling.
        raise NotImplementedError

    def _get_or_create_rho_pipeline_id(self) -> int:
        raise NotImplementedError

    def _prepare_holdout_set(
        self, next_trigger_id: int, rho_pipeline_id: int, selector_storage_backend: AbstractStorageBackend
    ) -> None:
        current_trigger_dataset_size = get_trigger_dataset_size(
            selector_storage_backend, self._pipeline_id, next_trigger_id, tail_triggers=0
        )

        holdout_set_size = min(int(current_trigger_dataset_size * self.holdout_set_ratio / 100), 1)

        stmt = self._get_holdout_sampling_query(self._pipeline_id, next_trigger_id, holdout_set_size).execution_options(
            yield_per=self.maximum_keys_in_memory
        )

        def training_set_producer() -> Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]:
            with MetadataDatabaseConnection(self._modyn_config) as database:
                for chunk in database.session.execute(stmt).partitions():
                    samples = [res[0] for res in chunk]
                    random.shuffle(samples)
                    yield [(sample, 1.0) for sample in samples], {}

        AbstractSelectionStrategy.store_training_set(
            rho_pipeline_id,
            next_trigger_id,
            self._modyn_config,
            training_set_producer,
            selector_storage_backend.insertion_threads,
        )
        raise NotImplementedError

    @staticmethod
    def _get_holdout_sampling_query(main_pipeline_id: int, trigger_id: int, target_size: int) -> Select:
        subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == main_pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == trigger_id,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )

        stmt = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == main_pipeline_id,
                SelectorStateMetadata.sample_key.in_(subq),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        return stmt
