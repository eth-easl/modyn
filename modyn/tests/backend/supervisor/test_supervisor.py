from modyn.backend.supervisor import Supervisor


def test_initialization() -> None:
    test = Supervisor({}, {}, None)  # noqa: F841 # pylint: disable=unused-variable
