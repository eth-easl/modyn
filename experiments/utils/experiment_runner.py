import logging
from concurrent.futures import ThreadPoolExecutor

from modyn.config.schema.pipeline import ModynPipelineConfig
from modynclient.client.client import Client
from modynclient.config.schema.client_config import ModynClientConfig

logger = logging.getLogger(__name__)


def run_multiple_pipelines(
    client_config: ModynClientConfig,
    pipeline_configs: list[ModynPipelineConfig],
    start_replay_at: int = 0,
    stop_replay_at: int | None = None,
    maximum_triggers: int | None = None,
    show_eval_progress: bool = True,
) -> None:
    logger.info("Start running multiple experiments!")

    for pipeline_config in pipeline_configs:
        client = Client(
            client_config, pipeline_config.model_dump(by_alias=True), start_replay_at, stop_replay_at, maximum_triggers
        )
        logger.info(f"Starting pipeline: {pipeline_config.pipeline.name}")
        started = client.start_pipeline()
        result = False
        if started:
            result = client.poll_pipeline_status(show_eval_progress=show_eval_progress)
        logger.info(f"Finished pipeline: {pipeline_config.pipeline.name}")

        if not result:
            logger.error("Client exited with error, aborting.")
            return

    logger.info("Finished running multiple experiments!")


def run_multiple_pipelines_parallel(
    client_config: ModynClientConfig,
    pipeline_configs: list[ModynPipelineConfig],
    start_replay_at: int,
    stop_replay_at: int | None,
    maximum_triggers: int | None,
    show_eval_progress: bool,
    maximal_collocation: int,
) -> None:
    logger.info("Start running multiple experiments in parallel!")
    # each time we take maximal_collocation pipelines and run them in parallel
    for i in range(0, len(pipeline_configs), maximal_collocation):
        with ThreadPoolExecutor(max_workers=maximal_collocation) as executor:
            executor.submit(
                run_multiple_pipelines,
                client_config,
                pipeline_configs[i : i + maximal_collocation],
                start_replay_at,
                stop_replay_at,
                maximum_triggers,
                show_eval_progress,
            )
    logger.info("Finished running multiple experiments in parallel!")
