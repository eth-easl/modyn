# pylint: disable=abstract-class-instantiated,unused-argument
from modyn.backend.supervisor.internal.trigger import Trigger
from unittest.mock import patch, MagicMock
from typing import Callable
import pytest


@patch.multiple(Trigger, __abstractmethods__=set())
def get_trigger(callback: Callable) -> Trigger:
    return Trigger(callback, {})


def test_initialization() -> None:
    def callback():
        pass

    trigger = get_trigger(callback)

    assert trigger.callback == callback  # pylint: disable=comparison-with-callable


@patch.object(Trigger, '_decide_for_trigger', return_value=0)
def test_inform_does_nothing_on_no_trigger(test__decide_for_trigger) -> None:
    callback = MagicMock()

    trigger = get_trigger(callback)
    assert not trigger.inform([])

    callback.assert_not_called()


@pytest.mark.parametrize("num_callbacks", [1, 2, 3, 4, 5])
def test_inform_calls_callback(num_callbacks) -> None:
    with patch.object(Trigger, '_decide_for_trigger', return_value=num_callbacks):
        callback = MagicMock()

        trigger = get_trigger(callback)
        assert trigger.inform([])
        assert callback.call_count == num_callbacks
