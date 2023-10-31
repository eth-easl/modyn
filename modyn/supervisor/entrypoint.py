import argparse
import logging
import pathlib
from typing import Any

import yaml
from modyn.supervisor.internal.grpc.supervisor_grpc_server import SupervisorGRPCServer

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Training Supervisor")
    parser_.add_argument(
        "config",
        type=pathlib.Path,
        action="store",
        help="Modyn infrastructure configuration file",
    )
    return parser_


def validate_args(args: Any) -> None:
    assert args.config.is_file(), f"File does not exist: {args.config}"


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    validate_args(args)

    with open(args.config, "r", encoding="utf-8") as config_file:
        modyn_config = yaml.safe_load(config_file)

    logger.info("Initializing supervisor.")
    with SupervisorGRPCServer(modyn_config):
        pass


if __name__ == "__main__":
    main()
