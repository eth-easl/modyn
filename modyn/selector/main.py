from pathlib import Path
from typing import Annotated

import typer
from modyn.config import read_modyn_config
from modyn.selector.internal.grpc.selector_server import SelectorGRPCServer
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(config: Annotated[Path, typer.Argument(help="Modyn infrastructure configuration file")]) -> None:
    """Entrypoint for the selector service."""
    modyn_config = read_modyn_config(config)

    logger.info("Initializing selector server.")
    with SelectorGRPCServer(modyn_config.model_dump(by_alias=True)):
        pass

    logger.info("Selector server returned, exiting.")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
