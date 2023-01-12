import argparse
import logging
import pathlib

import yaml
from modyn.backend.selector.selector_server import SelectorServer

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Selector")
    parser_.add_argument(
        "config",
        type=pathlib.Path,
        action="store",
        help="Modyn infrastructure configuration file",
    )
    parser_.add_argument(
        "pipeline",
        type=pathlib.Path,
        action="store",
        help="Modyn pipeline configuration file",
    )

    return parser_


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.config, "r", encoding="utf-8") as config_file:
        modyn_config = yaml.safe_load(config_file)

    with open(args.pipeline, "r", encoding="utf-8") as pipeline_file:
        pipeline_config = yaml.safe_load(pipeline_file)

    logger.info("Initializing selector.")
    selector = SelectorServer(pipeline_config, modyn_config)
    logger.info("Starting selector.")
    selector.run()

    logger.info("Selector returned, exiting.")


if __name__ == "__main__":
    main()
