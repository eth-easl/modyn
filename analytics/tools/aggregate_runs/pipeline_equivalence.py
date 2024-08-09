from copy import deepcopy

from modyn.config.schema.pipeline.sampling.config import CoresetStrategyConfig
from modyn.config.schema.pipeline.sampling.downsampling_config import (
    RHOLossDownsamplingConfig,
)
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs


def assert_pipeline_equivalence(logs: list[PipelineLogs]) -> None:
    # assert that all pipelines are the same except from the seed
    assert len(logs) >= 1

    candidates = [deepcopy(log) for log in logs]
    # set seeds to seed of first pipeline
    # set device to first pipeline since that does not matter
    for candidate in candidates:
        assert candidate.config.pipeline.evaluation
        assert candidates[0].config.pipeline.evaluation

        candidate.config.pipeline.training.seed = candidates[0].config.pipeline.training.seed
        candidate.config.pipeline.training.device = candidates[0].config.pipeline.training.device
        candidate.config.pipeline.evaluation.device = candidates[0].config.pipeline.evaluation.device
        candidate.config.pipeline.evaluation.after_pipeline_evaluation_workers = candidates[
            0
        ].config.pipeline.evaluation.after_pipeline_evaluation_workers
        candidate.config.pipeline.evaluation.after_training_evaluation_workers = candidates[
            0
        ].config.pipeline.evaluation.after_training_evaluation_workers

        if isinstance(candidate.config.pipeline.selection_strategy, CoresetStrategyConfig) and isinstance(
            candidate.config.pipeline.selection_strategy.downsampling_config,
            RHOLossDownsamplingConfig,
        ):
            candidate.config.pipeline.selection_strategy.downsampling_config.il_training_config.device = candidates[
                0
            ].config.pipeline.selection_strategy.downsampling_config.il_training_config.device
            candidate.config.pipeline.selection_strategy.downsampling_config.il_training_config.seed = candidates[
                0
            ].config.pipeline.selection_strategy.downsampling_config.il_training_config.seed

    assert all(
        [candidate.config == candidates[0].config for candidate in candidates]
    ), "Not all pipelines are the same (ignoring seed)"
