import argparse
import logging
import pathlib

import yaml
from modyn.backend.supervisor import Supervisor

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Training Supervisor")
    parser_.add_argument("pipeline", type=pathlib.Path, action="store", help="Pipeline configuration file")
    parser_.add_argument("config", type=pathlib.Path, action="store", help="Modyn infrastructure configuration file")

    parser_.add_argument(
        "--start-replay-at",
        type=int,
        action="store",
        help="This mode does not trigger on new data but just "
        "replays data starting at `TIMESTAMP` and ends all training afterwards. "
        "`TIMESTAMP` can be 0 and then just replays all data. See README for more.",
    )

    return parser_


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.pipeline.is_file(), f"File does not exist: {args.pipeline}"
    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.pipeline, "r", encoding="utf-8") as pipeline_file:
        pipeline_config = yaml.safe_load(pipeline_file)

    with open(args.config, "r", encoding="utf-8") as config_file:
        modyn_config = yaml.safe_load(config_file)

    if args.start_replay_at is not None:
        logger.info(f"Starting supervisor in experiment mode. Replay timestamp is set to {args.start_replay_at}")

    logger.info("Initializing supervisor.")
    supervisor = Supervisor(pipeline_config, modyn_config, args.start_replay_at)
    logger.info("Starting pipeline.")
    supervisor.pipeline()

    logger.info("Supervisor returned, exiting.")


if __name__ == "__main__":
    main()
