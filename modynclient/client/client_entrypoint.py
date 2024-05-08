import argparse
import logging
import pathlib
from typing import Any

import yaml
from modynclient.client import Client
from modynclient.config import read_client_config

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Training Supervisor")
    parser_.add_argument(
        "pipeline",
        type=pathlib.Path,
        action="store",
        help="Pipeline configuration file",
    )
    parser_.add_argument(
        "config",
        type=pathlib.Path,
        action="store",
        help="Modyn Client configuration file",
    )

    parser_.add_argument(
        "--start-replay-at",
        type=int,
        action="store",
        help="Enables experiment mode.  "
        "replays data newer or equal to `TIMESTAMP` and ends pipeline afterwards. "
        "`TIMESTAMP` can be 0 and then just replays all data.",
    )

    parser_.add_argument(
        "--stop-replay-at",
        type=int,
        action="store",
        help="Optional addition to `start-replay-at`. Defines the end of the replay interval (inclusive). "
        "If not given, all data is replayed.",
    )

    parser_.add_argument(
        "--maximum-triggers",
        type=int,
        action="store",
        help="Optional parameter to limit the maximum number of triggers that we consider before exiting.",
    )

    parser_.add_argument(
        "--evaluation-matrix",
        action="store_true",
        help="Whether to build an evaluation matrix of all models/triggers at the end. Currently just for "
        "experiments, does not overlap training and evaluation.",
    )

    return parser_


def validate_args(args: Any) -> None:
    assert args.pipeline.is_file(), f"File does not exist: {args.pipeline}"
    assert args.config.is_file(), f"File does not exist: {args.config}"

    if args.start_replay_at is None and args.stop_replay_at is not None:
        raise ValueError(
            "--stop-replay-at was provided, but --start-replay-at was not."
        )

    if args.maximum_triggers is not None and args.maximum_triggers < 1:
        raise ValueError(f"maximum_triggers is {args.maximum_triggers}, needs to be >= 1")


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    validate_args(args)

    with open(args.pipeline, "r", encoding="utf-8") as pipeline_file:
        pipeline_config = yaml.safe_load(pipeline_file)

    client_config = read_client_config(args.config)

    if args.start_replay_at is not None:
        logger.info(
            f"Starting client in experiment mode. Starting replay at {args.start_replay_at}."
        )
        if args.stop_replay_at is not None:
            logger.info(f"Replay interval ends at {args.stop_replay_at}.")

    logger.info("Initializing client.")
    logger.info(f"{client_config.model_dump()}")
    client = Client(
        client_config,
        pipeline_config,
        args.start_replay_at,
        args.stop_replay_at,
        args.maximum_triggers,
    )

    logger.info("Starting pipeline.")
    started = client.start_pipeline()
    if started:
        client.poll_pipeline_status()

    logger.info("Client returned, exiting.")


if __name__ == "__main__":
    main()
