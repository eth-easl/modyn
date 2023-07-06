"""Entrypoint for the evaluator service."""

import argparse
import logging
import pathlib

import yaml
from modyn.evaluator.evaluator import Evaluator

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        argparse.ArgumentParser: Argument parser
    """
    parser_ = argparse.ArgumentParser(description="Modyn Evaluator")
    parser_.add_argument("config", type=pathlib.Path, action="store", help="Modyn infrastructure configuration file")

    return parser_


def main() -> None:
    """Entrypoint for the evaluator service."""
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.config, "r", encoding="utf-8") as config_file:
        modyn_config = yaml.safe_load(config_file)

    logger.info("Initializing evaluator.")
    evaluator = Evaluator(modyn_config)
    logger.info("Starting evaluator.")
    evaluator.run()

    logger.info("Evaluator returned, exiting.")


if __name__ == "__main__":
    main()
