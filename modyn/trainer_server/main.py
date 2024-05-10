import pathlib
from typing import Annotated

import typer
from modyn.config import read_modyn_config
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(config: Annotated[pathlib.Path, typer.Argument(help="Modyn infrastructure configuration file")]) -> None:
    """Entrypoint for the modyn training server."""
    from modyn.trainer_server.trainer_server import TrainerServer  # pylint: disable=import-outside-toplevel

    modyn_config = read_modyn_config(config)

    logger.info("Initializing TrainerServer.")
    trainer_server = TrainerServer(modyn_config.model_dump(by_alias=True))
    logger.info("Starting TrainerServer.")
    trainer_server.run()

    logger.info("TrainerServer returned, exiting.")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
