from unittest.mock import MagicMock, patch

import pytest

from modyn.config.schema.pipeline.trigger import (
    AtLeastNEnsembleStrategy,
    DataAmountTriggerConfig,
    EnsembleTriggerConfig,
    TimeTriggerConfig,
)
from modyn.supervisor.internal.triggers import (
    DataAmountTrigger,
    EnsembleTrigger,
    TimeTrigger,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.models import TriggerPolicyEvaluationLog


@pytest.fixture
def ensemble_trigger_config() -> EnsembleTriggerConfig:
    return EnsembleTriggerConfig(
        subtriggers={
            "time_trigger": TimeTriggerConfig(every="5s"),
            "data_amount_trigger": DataAmountTriggerConfig(num_samples=8),
        },
        ensemble_strategy=AtLeastNEnsembleStrategy(n=2),
    )


@patch("modyn.supervisor.internal.triggers.timetrigger.TimeTrigger", autospec=True)
@patch("modyn.supervisor.internal.triggers.amounttrigger.DataAmountTrigger", autospec=True)
def test_initialization(
    mock_time_trigger: MagicMock,
    mock_data_amount_trigger: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    assert len(trigger.subtriggers) == 2
    assert isinstance(trigger.subtriggers["time_trigger"], TimeTrigger)
    assert isinstance(trigger.subtriggers["data_amount_trigger"], DataAmountTrigger)


@patch.object(DataAmountTrigger, "init_trigger")
@patch.object(TimeTrigger, "init_trigger")
def test_init_trigger(
    mock_time_trigger_init: MagicMock,
    mock_data_amount_init: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    mock_context = MagicMock(spec=TriggerContext)
    trigger.init_trigger(mock_context)

    # Ensure subtriggers are initialized with the context
    mock_time_trigger_init.assert_called_once_with(mock_context)
    mock_data_amount_init.assert_called_once_with(mock_context)


def test_reset_subtrigger_decision_cache(
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)

    subtrigger_decision_cache = {}
    trigger._reset_subtrigger_decision_cache(subtrigger_decision_cache)
    assert subtrigger_decision_cache == {
        "time_trigger": False,
        "data_amount_trigger": False,
    }

    subtrigger_decision_cache = {
        "time_trigger": True,
        "data_amount_trigger": False,
        "some_other_trigger": True,
    }
    trigger._reset_subtrigger_decision_cache(subtrigger_decision_cache)
    assert subtrigger_decision_cache == {
        "time_trigger": False,
        "data_amount_trigger": False,
    }


def test_update_next_subtrigger_index_cache(
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)

    subtrigger_generators = {
        "time_trigger": iter([4, 5, 12, 21]),
        "data_amount_trigger": iter([6, 20]),
    }
    next_subtrigger_index_cache = {
        "time_trigger": EnsembleTrigger.TRIGGER_SUBTRIGGER_NOT_TRIGGERED_YET,
        "data_amount_trigger": EnsembleTrigger.TRIGGER_SUBTRIGGER_NOT_TRIGGERED_YET,
    }

    # initial update
    trigger._update_next_subtrigger_index_cache(
        EnsembleTrigger.TRIGGER_SUBTRIGGER_NOT_TRIGGERED_YET,
        subtrigger_generators,
        next_subtrigger_index_cache,
    )
    assert next_subtrigger_index_cache == {
        "time_trigger": 4,
        "data_amount_trigger": 6,
    }

    # update one
    trigger._update_next_subtrigger_index_cache(4, subtrigger_generators, next_subtrigger_index_cache)
    assert next_subtrigger_index_cache == {
        "time_trigger": 5,
        "data_amount_trigger": 6,
    }

    # update one
    trigger._update_next_subtrigger_index_cache(5, subtrigger_generators, next_subtrigger_index_cache)
    assert next_subtrigger_index_cache == {
        "time_trigger": 12,
        "data_amount_trigger": 6,
    }

    # update both
    trigger._update_next_subtrigger_index_cache(14, subtrigger_generators, next_subtrigger_index_cache)
    assert next_subtrigger_index_cache == {
        "time_trigger": 21,
        "data_amount_trigger": 20,
    }

    # exhaust second generator, leave first untouched
    trigger._update_next_subtrigger_index_cache(20, subtrigger_generators, next_subtrigger_index_cache)
    assert next_subtrigger_index_cache == {
        "time_trigger": 21,
        "data_amount_trigger": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
    }

    # exhaust second
    trigger._update_next_subtrigger_index_cache(21, subtrigger_generators, next_subtrigger_index_cache)
    assert next_subtrigger_index_cache == {
        "time_trigger": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "data_amount_trigger": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
    }


def test_find_next_trigger_index(
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)

    subtrigger_decision_cache = {"t1": False, "t2": False, "t3": False, "t4": False}
    next_subtrigger_index_cache = {
        "t1": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t2": 10,
        "t3": 5,
        "t4": 3,
    }

    trigger.config.ensemble_strategy = AtLeastNEnsembleStrategy(n=2)

    new_data = [(0, 0, 0)]  # dummy

    trigger_log = TriggerPolicyEvaluationLog()

    # trigger after second subtrigger
    result = trigger._find_next_trigger_index(
        processing_head=0,
        new_data=new_data,
        subtrigger_decision_cache=subtrigger_decision_cache,
        next_subtrigger_index_cache=next_subtrigger_index_cache,
        log=trigger_log,
    )

    assert result == 5
    assert len(trigger_log.evaluations) == 2
    assert [e.trigger_index for e in trigger_log.evaluations] == [3, 5]
    assert [e.triggered for e in trigger_log.evaluations] == [False, True]

    trigger_log = TriggerPolicyEvaluationLog()

    # no trigger case, not 3 subtriggers
    subtrigger_decision_cache = {"t1": False, "t2": False, "t3": False, "t4": False}
    next_subtrigger_index_cache = {
        "t1": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t2": 10,
        "t3": 5,
        "t4": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
    }
    trigger.config.ensemble_strategy = AtLeastNEnsembleStrategy(n=3)
    result = trigger._find_next_trigger_index(
        processing_head=0,
        new_data=new_data,
        subtrigger_decision_cache=subtrigger_decision_cache,
        next_subtrigger_index_cache=next_subtrigger_index_cache,
        log=trigger_log,
    )
    assert result is None
    assert len(trigger_log.evaluations) == 2
    assert [e.trigger_index for e in trigger_log.evaluations] == [5, 10]
    assert [e.triggered for e in trigger_log.evaluations] == [False, False]

    trigger_log = TriggerPolicyEvaluationLog()

    # no subtrigger
    subtrigger_decision_cache = {"t1": False, "t2": False, "t3": False, "t4": False}
    next_subtrigger_index_cache = {
        t: EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED for t in ["t1", "t2", "t3", "t4"]
    }
    result = trigger._find_next_trigger_index(
        processing_head=0,
        new_data=new_data,
        subtrigger_decision_cache=subtrigger_decision_cache,
        next_subtrigger_index_cache=next_subtrigger_index_cache,
        log=trigger_log,
    )
    assert result is None
    assert len(trigger_log.evaluations) == 0

    # Test that previous batch's decision is used
    subtrigger_decision_cache = {"t1": True, "t2": True, "t3": False, "t4": False}
    next_subtrigger_index_cache = {
        "t1": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t2": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t3": 10,
        "t4": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
    }
    trigger.config.ensemble_strategy = AtLeastNEnsembleStrategy(n=3)
    result = trigger._find_next_trigger_index(
        processing_head=0,
        new_data=new_data,
        subtrigger_decision_cache=subtrigger_decision_cache,
        next_subtrigger_index_cache=next_subtrigger_index_cache,
        log=trigger_log,
    )
    assert result == 10
    assert len(trigger_log.evaluations) == 1

    # Test that previous batch's decision is used, but not sufficient if no new trigger is found
    subtrigger_decision_cache = {"t1": True, "t2": True, "t3": False, "t4": False}
    # t1 is already triggered
    next_subtrigger_index_cache = {
        "t1": 10,
        "t2": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t3": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
        "t4": EnsembleTrigger.TRIGGER_SUBTRIGGER_GENERATOR_EXHAUSTED,
    }
    trigger_log = TriggerPolicyEvaluationLog()
    result = trigger._find_next_trigger_index(
        processing_head=0,
        new_data=new_data,
        subtrigger_decision_cache=subtrigger_decision_cache,
        next_subtrigger_index_cache=next_subtrigger_index_cache,
        log=trigger_log,
    )
    assert result is None
    assert len(trigger_log.evaluations) == 1


def test_inform_trigger(ensemble_trigger_config: EnsembleTriggerConfig) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)

    # Run the inform method: 2 batches --> 2 calls to each subtrigger's inform method
    trigger_log = TriggerPolicyEvaluationLog()
    results = list(trigger.inform([(i, i + 100, 1) for i in range(20)], log=trigger_log))

    # timetrigger at 4, 9, 14
    # dataamounttrigger at 7, 15

    assert len(results) == 2
    assert results == [7, 15]

    # index 14 of timetrigger is ignored as the timetrigger is already in triggered state until the
    # dataamounttrigger triggers at 15

    assert len(trigger_log.evaluations) == 4
    assert [e.trigger_index for e in trigger_log.evaluations] == [4, 7, 9, 15]

    assert trigger_log.evaluations[0].trigger_index == 4
    assert not trigger_log.evaluations[0].triggered

    assert trigger_log.evaluations[1].trigger_index == 7
    assert trigger_log.evaluations[1].triggered

    assert trigger_log.evaluations[2].trigger_index == 9
    assert not trigger_log.evaluations[2].triggered

    assert trigger_log.evaluations[3].trigger_index == 15
    assert trigger_log.evaluations[3].triggered


def test_inform_multi_batch(ensemble_trigger_config: EnsembleTriggerConfig) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)

    # Run the inform method: 2 batches --> 2 calls to each subtrigger's inform method
    trigger_log = TriggerPolicyEvaluationLog()

    # timetrigger at 4, 9, ...
    # dataamounttrigger at 7, ...

    # first inform didn't contain 7
    results = list(trigger.inform([(i, i + 100, 1) for i in range(6)], log=trigger_log))

    assert len(results) == 0
    assert len(trigger_log.evaluations) == 1

    trigger_log = TriggerPolicyEvaluationLog()

    # second inform contains 7, timetrigger remains in triggered state from index 4
    results = list(trigger.inform([(i, i + 100, 1) for i in range(6, 9)], log=trigger_log))

    assert len(results) == 1
    assert results == [1]  # = element 7
    assert len(trigger_log.evaluations) == 1


@patch.object(DataAmountTrigger, "inform_new_model")
@patch.object(TimeTrigger, "inform_new_model")
def test_inform_new_model(
    mock_time_trigger_inform_new_model: MagicMock,
    mock_data_amount_inform_new_model: MagicMock,
    ensemble_trigger_config: EnsembleTriggerConfig,
) -> None:
    trigger = EnsembleTrigger(ensemble_trigger_config)
    mock_model_id = 99
    trigger.inform_new_model(mock_model_id)
    mock_time_trigger_inform_new_model.assert_called_once_with(mock_model_id)
    mock_data_amount_inform_new_model.assert_called_once_with(mock_model_id)
