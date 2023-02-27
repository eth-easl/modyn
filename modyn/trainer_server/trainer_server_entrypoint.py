import argparse
import logging
import multiprocessing as mp
import pathlib

import yaml

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)

# We need to do this at the top because the FTP Server or other dependencies otherwise set fork.
try:
    mp.set_start_method("spawn")
except RuntimeError as error:
    if mp.get_start_method() != "spawn":
        logger.error("Start method is already set to {}", mp.get_start_method())
        raise error

from modyn.trainer_server.trainer_server import TrainerServer  # noqa # pylint: disable=wrong-import-position


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Training Server")
    parser_.add_argument(
        "config",
        type=pathlib.Path,
        action="store",
        help="Modyn infrastructure configuration file",
    )

    return parser_


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    logger.info("Initializing TrainerServer.")
    trainer_server = TrainerServer(config)
    logger.info("Starting TrainerServer.")
    trainer_server.run()

    logger.info("TrainerServer returned, exiting.")


if __name__ == "__main__":
    main()
