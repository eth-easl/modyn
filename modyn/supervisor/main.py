from pathlib import Path
from typing import Annotated

import typer
from modyn.config import read_modyn_config
from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(config: Annotated[Path, typer.Argument(help="Modyn infrastructure configuration file")]) -> None:
    "Modyn Training Supervisor"
    modyn_config = read_modyn_config(config)

    logger.info("Initializing supervisor server.")
    with SupervisorGRPCServer(modyn_config.model_dump(by_alias=True)):
        pass

    logger.info("Supervisor server returned, exiting.")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
