import pathlib

from modyn.supervisor.internal.evaluation_result_writer import JsonResultWriter


def test_init():
    writer = JsonResultWriter(10, 15, pathlib.Path(""))
    assert writer.pipeline_id == 10
    assert writer.trigger_id == 15
    assert str(writer.eval_directory) == "."
