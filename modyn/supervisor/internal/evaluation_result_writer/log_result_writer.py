# pylint: disable=no-name-in-module
from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter


class LogResultWriter(JsonResultWriter):
    def store_results(self) -> None:
        pass
