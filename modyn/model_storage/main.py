"""Entrypoint for the model storage service."""

from pathlib import Path
from typing import Annotated

import typer
from modyn.config import read_modyn_config
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(config: Annotated[Path, typer.Argument(help="Modyn infrastructure configuration file")]) -> None:
    """Entrypoint for the model storage service."""
    from modyn.model_storage.model_storage import ModelStorage  # pylint: disable=import-outside-toplevel

    modyn_config = read_modyn_config(config)

    logger.info("Initializing model storage.")
    model_storage = ModelStorage(modyn_config.model_dump(by_alias=True))

    logger.info("Starting model storage.")
    model_storage.run()

    logger.info("Shutting down model storage.")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
