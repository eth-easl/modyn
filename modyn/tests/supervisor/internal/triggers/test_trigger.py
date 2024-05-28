# pylint: disable=abstract-class-instantiated,unused-argument
from unittest.mock import patch

from modyn.supervisor.internal.triggers import Trigger


@patch.multiple(Trigger, __abstractmethods__=set())
def test_initialization() -> None:
    _ = Trigger()
