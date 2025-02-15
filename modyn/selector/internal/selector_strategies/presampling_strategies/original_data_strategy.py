from sqlalchemy import Select, asc, func, select

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend


class OriginalSetPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self.requires_trigger_dataset_size = True
        self.first_trigger_ratio = presampling_config.get("first_trigger_ratio", 0.5)

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ) -> Select:
        """
        - In `trigger_id = 0`: Select **ALL** available data.
        - In later triggers (`next_trigger_id > 0`): Select:
          1. A random fraction (`first_trigger_ratio`) of samples from `trigger_id = 0`.
          2. All samples from the current trigger (`next_trigger_id`).
        """
        assert trigger_dataset_size is not None
        assert trigger_dataset_size >= 0

        if next_trigger_id == 0:
            stmt = (
                select(SelectorStateMetadata.sample_key)
                .filter(SelectorStateMetadata.pipeline_id == self.pipeline_id)
                .order_by(asc(SelectorStateMetadata.timestamp))
            )
            return stmt

        first_trigger_subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == 0,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(int(trigger_dataset_size * self.first_trigger_ratio))
        )

        current_trigger_subq = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.pipeline_id == self.pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id == next_trigger_id,  # 🔹 Select ALL from the current trigger
        )

        stmt = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.sample_key.in_(first_trigger_subq.union(current_trigger_subq)),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )
        return stmt
