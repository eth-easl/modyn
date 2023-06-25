from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from sqlalchemy import Select, asc, select


class AllDataPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        config: dict,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
        tail_triggers: int,
        has_limit: bool,
        training_set_size_limit: int,
    ):
        super().__init__(
            config, modyn_config, pipeline_id, maximum_keys_in_memory, tail_triggers, has_limit, training_set_size_limit
        )

        if "presampling_ratio" in config and config["presampling_ratio"] != 100:
            raise ValueError("Using AllDataPresamplingStrategy, the presampling_ratio is implicitly 100%")

    def get_query_stmt(self, next_trigger_id: int) -> Select:
        stmt = (
            select(SelectorStateMetadata.sample_key)
            # Enables batching of results in chunks.
            # See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
            .execution_options(yield_per=self._maximum_keys_in_memory)
            .filter(
                SelectorStateMetadata.pipeline_id == self._pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - self._tail_triggers
                if self._tail_triggers is not None
                else True,
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        if self._has_limit:
            stmt = stmt.limit(self._training_set_size_limit)

        return stmt
