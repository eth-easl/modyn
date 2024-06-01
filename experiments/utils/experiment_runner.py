import logging

from modyn.config.schema.pipeline.pipeline import ModynPipelineConfig
from modynclient.client.client import Client
from modynclient.config.schema.client_config import ModynClientConfig

logger = logging.getLogger(__name__)


def run_multiple_pipelines(
    client_config: ModynClientConfig,
    pipeline_configs: list[ModynPipelineConfig],
    start_replay_at: int = 0,
    stop_replay_at: int | None = None,
    maximum_triggers: int | None = None,
) -> None:
    logger.info("Start running multiple experiments!")

    for pipeline_config in pipeline_configs:
        client = Client(
            client_config, pipeline_config.model_dump(by_alias=True), start_replay_at, stop_replay_at, maximum_triggers
        )
        logger.info(f"Starting pipeline: {pipeline_config.pipeline.name}")
        started = client.start_pipeline()
        if started:
            client.poll_pipeline_status()
        logger.info(f"Finished pipeline: {pipeline_config.pipeline.name}")

    logger.info("Finished running multiple experiments!")
