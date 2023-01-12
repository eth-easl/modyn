# pylint: disable=abstract-class-instantiated,unused-argument
from unittest.mock import patch

from modyn.backend.supervisor.internal.trigger import Trigger


@patch.multiple(Trigger, __abstractmethods__=set())
def get_trigger() -> Trigger:
    return Trigger({})


def test_initialization() -> None:
    _ = get_trigger()
