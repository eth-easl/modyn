from __future__ import annotations

import logging
from collections.abc import Generator
from typing import cast

from modyn.config.schema.pipeline.trigger.drift.config import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.ensemble import EnsembleTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.datadrifttrigger import DataDriftTrigger
from modyn.supervisor.internal.triggers.timetrigger import TimeTrigger
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.utils.models import (
    EnsembleTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)

logger = logging.getLogger(__name__)


class EnsembleTrigger(Trigger):
    """Evaluates multiple subtriggers and triggers if a certain after an
    aggregation function aggregates subtrigger decisions to True.

    Idea:
        The EnsembleTrigger always evaluates and caches the next triggering index
        from the generators of each subtrigger. The evaluations of all subsequent subtrigger indexes are
        deferred in a lazy manner.

        We assume that after a subtrigger will always like to see a trigger after the next index in the generator.
        Thus, we are sure that we can trigger for certain at the maximum of all next trigger indexes of the subtriggers
        as all subtriggers requested a trigger until this index.

        For the potential triggering indexes up until this maximum, we evaluate the aggregation function at every
        index where one policy first decided to do a trigger. We have those indexes cached after fetching the next
        trigger index from the subtriggers.

        We have to ensure though that every trigger consumes / evaluates all data eventually as some triggers
        need the data to maintain their state.
        Therefore we have to ensure right after a trigger, that all next trigger indexes in the cache are
        in the future wr.t. our current trigger index. We can simply do that by fetching the next trigger indexes
        from the subtriggers until all subtriggers have a next trigger index in the future.
    """

    def __init__(self, config: EnsembleTriggerConfig):
        self.config = config
        self.subtriggers = self._create_subtriggers(config)

        self.last_inform_decisions: dict[str, bool] = {}
        """Leftover decisions from the last inform call of the subtriggers that
        didn't yield a trigger."""

    def init_trigger(self, context: TriggerContext) -> None:
        for trigger in self.subtriggers.values():
            trigger.init_trigger(context)

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        subtrigger_generators = {
            trigger_name: trigger.inform(new_data, log) for trigger_name, trigger in self.subtriggers.items()
        }
        subtrigger_decision_cache = dict(
            {trigger_name: False for trigger_name in self.subtriggers.keys()},
            **self.last_inform_decisions,
        )
        """Indicates whether the ensemble trigger has passed a triggering index
        with processing_head for a subtrigger since the last trigger.

        Will be reset after every trigger.
        """

        next_subtrigger_index_cache: dict[str, int | None] = {
            trigger_name: -1 for trigger_name in self.subtriggers.keys()
        }
        """Cache for the next trigger index of each subtrigger.

        None indicates that the subtrigger has no more triggers because the generator is exhausted.
        -1 indicates that the last triggering index was in the last batch or there was no trigger yet.
        """

        processing_head = -1
        """The current candidate index for the next trigger.

        If the aggregation function returns True, we trigger. If not,
        the processing head is still increased to the next triggering
        index of a subtrigger. Indexes are based on the new_data batch.
        """

        self._update_outdated_next_trigger_indexes(
            processing_head,
            subtrigger_generators,
            next_subtrigger_index_cache,
        )

        while True:
            next_trigger_idx = self._find_next_trigger_index(
                processing_head,
                new_data,
                subtrigger_generators,
                subtrigger_decision_cache,
                next_subtrigger_index_cache,
                log,
            )
            if next_trigger_idx is None:
                # complete remaining inform calls regardless of the decision, any of the decision will lead to a
                # non triggering aggregation result

                break

            processing_head = next_trigger_idx

            # reset state for next detection pass;
            self._reset_subtrigger_decision_cache(subtrigger_decision_cache)
            self._update_outdated_next_trigger_indexes(
                processing_head,
                subtrigger_generators,
                next_subtrigger_index_cache,
            )

            yield next_trigger_idx

        self.last_inform_decisions = subtrigger_decision_cache

    def inform_new_model(self, most_recent_model_id: int) -> None:
        for trigger in self.subtriggers.values():
            trigger.inform_new_model(most_recent_model_id)

    # --------------------------------------------------- Internal --------------------------------------------------- #

    def _create_subtriggers(self, config: EnsembleTriggerConfig) -> dict[str, Trigger]:
        subtriggers: dict[str, Trigger] = {}
        for trigger_id, trigger_config in config.subtriggers.items():
            if isinstance(trigger_config, TimeTriggerConfig):
                subtriggers[trigger_id] = TimeTrigger(trigger_config)
            elif isinstance(trigger_config, DataAmountTriggerConfig):
                subtriggers[trigger_id] = DataAmountTrigger(trigger_config)
            elif isinstance(trigger_config, DataDriftTriggerConfig):
                subtriggers[trigger_id] = DataDriftTrigger(trigger_config)
            elif isinstance(trigger_config, EnsembleTriggerConfig):
                raise NotImplementedError("Nested ensemble triggers are not supported.")
            else:
                raise ValueError(f"Unknown trigger config: {trigger_config}")
        return subtriggers

    def _reset_subtrigger_decision_cache(self, subtrigger_decision_cache: dict[str, bool]) -> None:
        subtrigger_decision_cache.clear()
        subtrigger_decision_cache.update({trigger_name: False for trigger_name in self.subtriggers.keys()})

    def _update_outdated_next_trigger_indexes(
        self,
        processing_head: int,
        subtrigger_generators: dict[str, Generator[int, None, None]],
        next_subtrigger_index_cache: dict[str, int | None],
    ) -> None:
        """Updates the next trigger indexes in the cache to be in the future
        w.r.t. the current trigger index.

        Supposed to be called after a new trigger index was found or at the beginning of the inform method to
        sync. with the new data batch.

        Note: Arguments are modified in place.
        """
        for trigger_name, generator in subtrigger_generators.items():
            # generator is not exhausted
            if next_subtrigger_index_cache[trigger_name] is not None:
                try:
                    while cast(dict[str, int], next_subtrigger_index_cache)[trigger_name] <= processing_head:
                        next_subtrigger_index_cache[trigger_name] = next(generator)
                except StopIteration:
                    next_subtrigger_index_cache[trigger_name] = None

    def _find_next_trigger_index(
        self,
        processing_head: int,
        new_data: list[tuple[int, int, int]],
        subtrigger_generators: dict[str, Generator[int, None, None]],
        subtrigger_decision_cache: dict[str, bool],
        next_subtrigger_index_cache: dict[str, int | None],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> int | None:
        """Fetches the next trigger index from the subtriggers and caches
        updates internal state via side effects.

        Returns:
            The next trigger index or None if no subtrigger has a next trigger index.
            None indicates that even if all unexhausted subtriggers would trigger, the ensemble trigger would not
            trigger. This indicates that the batch is completed and no trigger will be emitted anymore. The remaining
            inform calls to triggering subtriggers are still necessary to maintain their state.

        Note: Arguments are modified in place.
        """
        unfinished_index_cache = {
            trigger_name: generator
            for trigger_name, generator in next_subtrigger_index_cache.items()
            if next_subtrigger_index_cache[trigger_name] is not None
        }
        if len(unfinished_index_cache) == 0:
            return None

        # assert that all subtrigger indexes are in the future
        assert all(
            cast(dict[str, int], unfinished_index_cache)[trigger_name] > processing_head
            for trigger_name in unfinished_index_cache.keys()
        ), "Subtrigger indexes are not in the future."

        # find next trigger candidate: use the minimum of all next trigger indexes;
        # sort subtriggers by next trigger index
        next_trigger_candidates = sorted(
            cast(dict[str, int], unfinished_index_cache).items(),
            key=lambda item: item[1],
        )

        for subtrigger_name, next_subtrigger_index in next_trigger_candidates:
            processing_head = next_subtrigger_index
            subtrigger_decision_cache[subtrigger_name] = True

            # evaluate trigger aggregator function
            aggregation_result = self.config.ensemble_strategy.aggregate_decision_func(subtrigger_decision_cache)

            if log:
                ensemble_eval_log = EnsembleTriggerEvalLog(
                    triggered=aggregation_result,
                    trigger_index=processing_head,
                    evaluation_interval=(new_data[0][0:2], new_data[-1][0:2]),
                    subtrigger_decisions=dict(subtrigger_decision_cache),
                )
                log.evaluations.append(ensemble_eval_log)

            if aggregation_result:
                return processing_head

        return None
