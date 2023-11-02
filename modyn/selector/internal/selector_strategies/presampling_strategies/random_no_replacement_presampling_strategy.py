from typing import Optional

from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend
from modyn.selector.internal.storage_backend.database.database_storage_backend import DatabaseStorageBackend
from sqlalchemy import Select, asc, func, select
from sqlalchemy.orm.session import Session


class RandomNoReplacementPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self, presampling_config: dict, modyn_config: dict, pipeline_id: int, storage_backend: AbstractStorageBackend
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self.requires_trigger_dataset_size = True

        # a complete_trigger is the last time when all the datapoints (in the db at that time) have been seen
        self.last_complete_trigger = 0

        # TODO(create issue): Support local backend on AbstractBalancedStrategy
        assert isinstance(
            self._storage_backend, DatabaseStorageBackend
        ), "AbstractBalancedPresamplingStrategy currently only supports the DatabaseStorageBackend"

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
    ) -> Select:
        assert trigger_dataset_size is not None
        assert trigger_dataset_size >= 0

        target_size = self.get_target_size(trigger_dataset_size, limit)

        # get the query to sample up to target size points that have not been used since last_complete_trigger
        subq = self._get_query_sample_random(next_trigger_id, tail_triggers, target_size)

        # update last_used_in_trigger of the samples selected in the subquery
        self._update_last_used_in_trigger(next_trigger_id, subq)

        # count how many samples are retrieved
        number_of_sampled_points = self._count_number_of_sampled_points(next_trigger_id)
        if number_of_sampled_points < target_size:
            # we have used all the available samples so we have to update the last_complete_trigger
            self.last_complete_trigger = next_trigger_id
            remaining_points_to_be_sampled = target_size - number_of_sampled_points

            # repeat the above for remaining_points_to_be_sampled. Note that now self.last_complete_trigger is changed,
            # so points that were not sampled before can now be taken
            subq = self._get_query_sample_random(next_trigger_id, tail_triggers, remaining_points_to_be_sampled)
            self._update_last_used_in_trigger(next_trigger_id, subq)

        # then the query to select the samples is straightforward, just a filter on last_used_in_trigger
        stmt = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.last_used_in_trigger == next_trigger_id,
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        return stmt

    def _get_query_sample_random(self, next_trigger_id: int, tail_triggers: Optional[int], target_size: int) -> Select:
        subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
                if tail_triggers is not None
                else True,
                # just consider points that have not been used since the last complete trigger
                SelectorStateMetadata.last_used_in_trigger < self.last_complete_trigger,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )
        return subq

    def _update_last_used_in_trigger(self, next_trigger_id: int, subq: Select) -> None:
        def _session_callback(session: Session) -> None:
            session.query(SelectorStateMetadata).filter(
                SelectorStateMetadata.pipeline_id == self.pipeline_id,
                SelectorStateMetadata.sample_key.in_(subq),
            ).update({"last_used_in_trigger": next_trigger_id})
            session.commit()

        self._storage_backend._execute_on_session(_session_callback)

    def _count_number_of_sampled_points(self, next_trigger_id: int) -> int:
        def _session_callback(session: Session) -> None:
            return (
                session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self.pipeline_id,
                    SelectorStateMetadata.last_used_in_trigger == next_trigger_id,
                )
                .count()
            )

        return self._storage_backend._execute_on_session(_session_callback)
