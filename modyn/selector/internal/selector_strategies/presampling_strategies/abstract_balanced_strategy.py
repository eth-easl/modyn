from abc import ABC
from typing import Optional

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from sqlalchemy import Select, asc, func, select
from sqlalchemy.sql.expression import func as sql_func


def get_fair_share(capacity: int, requests: list[int]) -> int:
    """
    Simple Fair Sharing algorithm.
    Here it's used to determine how many samples gets each class/trigger (min(fair_share, num_samples)
    """
    fair_share = int(capacity / len(requests))
    below_fair_share = [requests.index(el) for el in requests if el <= fair_share]

    if len(below_fair_share) != 0 and len(requests) != len(below_fair_share):
        new_capacity = capacity - sum(requests[i] for i in below_fair_share)
        new_requests = [requests[i] for i in range(len(requests)) if i not in below_fair_share]
        return get_fair_share(new_capacity, new_requests)
    return fair_share


def get_fair_share_predicted_total(fair_share: int, requests: list[int]) -> int:
    return sum(min(fair_share, req) for req in requests)


class AbstractBalancedPresamplingStrategy(AbstractPresamplingStrategy, ABC):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(presampling_config, modyn_config, pipeline_id, maximum_keys_in_memory)

        self.force_required_target_size = presampling_config.get("force_required_target_size", False)
        self.force_column_balancing = presampling_config.get("force_column_balancing", False)
        self.balanced_column = None

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
        requires_samples_ordered_by_label: bool,
    ) -> Select:
        assert self.balanced_column is not None
        samples_count = self._get_samples_count_per_balanced_column(next_trigger_id, tail_triggers)
        target_size = self.get_target_size(sum(samples_count), limit)
        fair_share = get_fair_share(target_size, samples_count)

        # randomly permute samples class by class or trigger by trigger.
        # Then use row_number (in the next query) to get N random samples for each class/trigger
        subquery = select(
            SelectorStateMetadata,
            sql_func.row_number()
            .over(partition_by=self.balanced_column, order_by=func.random())  # pylint: disable=not-callable
            .label("row_num"),
        ).filter(
            SelectorStateMetadata.pipeline_id == self.pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
            if tail_triggers is not None
            else True,
        )

        if self.force_required_target_size:
            return self._get_force_required_target_size_query(fair_share, samples_count, subquery, target_size)
        if self.force_column_balancing:
            return self._get_force_column_balancing_query(fair_share, samples_count, subquery, target_size)

        return self._get_base_query(fair_share, subquery)

    def _get_force_column_balancing_query(
        self, fair_share: int, samples_count: list[int], subquery: Select, target_size: int
    ) -> Select:
        """
        Each class/trigger has exactly the same number of samples
        """
        smallest_size = min(samples_count)
        if smallest_size < fair_share:
            return (
                select(subquery.c.sample_key)
                .where(subquery.c.row_num <= smallest_size)
                .order_by(asc(subquery.c.timestamp))
                .limit(target_size)
            )
        return self._get_base_query(fair_share, subquery)

    def _get_base_query(self, fair_share: int, subquery: Select) -> Select:
        """

        Class/Trigger j gets min(fair_share, number_samples[j]) samples

        """
        return (
            select(subquery.c.sample_key)
            .execution_options(yield_per=self.maximum_keys_in_memory)
            .where(subquery.c.row_num <= fair_share)
            .order_by(asc(subquery.c.timestamp))
        )

    def _get_force_required_target_size_query(
        self, fair_share: int, samples_count: list[int], subquery: Select, target_size: int
    ) -> Select:
        """

        The returned number of samples is exactly target_size. Some classes/triggers might get more samples than others

        """
        predicted_number_of_samples = get_fair_share_predicted_total(fair_share, samples_count)
        if predicted_number_of_samples < target_size:
            # if we are below the target, overshoot and then limit
            return (
                select(subquery.c.sample_key)
                .where(subquery.c.row_num <= fair_share + 1)
                .order_by(asc(subquery.c.timestamp))
                .limit(target_size)
            )
        return self._get_base_query(fair_share, subquery)

    def _get_samples_count_per_balanced_column(self, next_trigger_id: int, tail_triggers: Optional[int]) -> list[int]:
        """

        Performs a group_by query and returns a list with the number of samples for each group

        """
        with MetadataDatabaseConnection(self.modyn_config) as database:
            query = (
                database.session.query(self.balanced_column, func.count())  # pylint: disable=not-callable
                .filter(
                    SelectorStateMetadata.pipeline_id == self.pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers
                    if tail_triggers is not None
                    else True,
                )
                .group_by(self.balanced_column)
            )
            samples_count = query.all()

        # el[0] is the class/trigger, el[1] is the count
        return [el[1] for el in samples_count]
