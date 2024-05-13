# pylint: disable=import-outside-toplevel

"""Entrypoint for the MetadataProcessor service."""

from pathlib import Path
from typing import Annotated

import typer
from modyn.config import read_modyn_config
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(config: Annotated[Path, typer.Argument(help="Modyn infrastructure configuration file")]) -> None:
    """Entrypoint for the metadata processor service."""
    from modyn.metadata_processor.internal.grpc.metadata_processor_server import MetadataProcessorServer

    modyn_config = read_modyn_config(config)

    logger.info("Initializing Metadata Processor")
    processor = MetadataProcessorServer(modyn_config.model_dump(by_alias=True))

    logger.info("Starting Metadata Processor")
    processor.run()

    logger.info("Processor returned, exiting")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
