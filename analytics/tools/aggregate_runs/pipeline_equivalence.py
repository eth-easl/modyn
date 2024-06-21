from copy import deepcopy

from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs


def assert_pipeline_equivalence(logs: list[PipelineLogs]) -> None:
    # assert that all pipelines are the same except from the seed
    assert len(logs) > 1

    candidates = [deepcopy(log) for log in logs]
    # set seeds to seed of first pipeline
    for i, candidate in enumerate(candidates):
        candidate.config.pipeline.training.seed = candidates[0].config.pipeline.training.seed

    assert all(
        [candidate.config == candidates[0].config for candidate in candidates]
    ), "Not all pipelines are the same (ignoring seed)"
