from pathlib import Path
from typing import Annotated, Optional

import typer
from modyn.config import read_pipeline
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn
from modynclient.client import Client
from modynclient.config import read_client_config

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def main(
    pipeline: Annotated[Path, typer.Argument(help="Pipeline configuration file")],
    config: Annotated[Path, typer.Argument(help="Modyn Client configuration file")],
    start_replay_at: Annotated[
        Optional[int],
        typer.Option(
            help=(
                "Enables experiment mode. Replays data newer or equal to `TIMESTAMP` and ends pipeline afterwards. "
                "`TIMESTAMP` can be 0 and then just replays all data."
            )
        ),
    ] = None,
    stop_replay_at: Annotated[
        Optional[int],
        typer.Option(
            help=(
                "Optional addition to `start-replay-at`. Defines the end of the replay interval (inclusive). "
                "If not given, all data is replayed."
            )
        ),
    ] = None,
    maximum_triggers: Annotated[
        Optional[int],
        typer.Option(
            help="Optional parameter to limit the maximum number of triggers that we consider before exiting."
        ),
    ] = None,
) -> None:
    # Validate arguments
    if start_replay_at is None and stop_replay_at is not None:
        raise ValueError(
            "--stop-replay-at was provided, but --start-replay-at was not."
        )

    if maximum_triggers is not None and maximum_triggers < 1:
        raise ValueError(f"maximum_triggers is {maximum_triggers}, needs to be >= 1")

    # Read files (includes validation)
    pipeline_config = read_pipeline(pipeline)
    client_config = read_client_config(config)

    if start_replay_at is not None:
        logger.info(
            f"Starting client in experiment mode. Starting replay at {start_replay_at}."
        )
        if stop_replay_at is not None:
            logger.info(f"Replay interval ends at {stop_replay_at}.")

    logger.info("Initializing client.")
    logger.info(f"{client_config}")
    client = Client(client_config, pipeline_config, start_replay_at, stop_replay_at, maximum_triggers)

    logger.info("Starting pipeline.")
    started = client.start_pipeline()
    if started:
        client.poll_pipeline_status()

    logger.info("Client returned, exiting.")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
