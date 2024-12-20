from sqlalchemy import Select, asc, select

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend


class NoPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
    ):
        if presampling_config.ratio != 100:
            raise ValueError("Using NoPresamplingStrategy, the presampling_ratio is implicitly 100%")
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ) -> Select:
        stmt = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                (
                    SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
                    if tail_triggers is not None
                    else True
                ),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        return stmt
