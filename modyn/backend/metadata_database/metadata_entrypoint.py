# TODO(vGsteiger): The next metadata database PR should add a
# `modyn-metadata` shell script similar to storage/supervisor
import argparse
import logging
import pathlib
from concurrent import futures

import grpc
import yaml
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import (  # noqa: E501, E402
    add_MetadataServicer_to_server,
)
from modyn.backend.metadata_database.internal.grpc.metadata_database_grpc_servicer import MetadataDatabaseGRPCServicer
from modyn.backend.metadata_database.metadata_database import MetadataDatabase

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Metadata Database")
    parser_.add_argument(
        "config",
        type=pathlib.Path,
        action="store",
        help="Modyn infrastructure configuration file",
    )
    return parser_


# TODO(vGsteiger): This not yet 100% aligned with the other components
# (encapsulation of servicer etc). This should be done in the next
# metadata database PR


def run(database: MetadataDatabase, config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MetadataServicer_to_server(MetadataDatabaseGRPCServicer(database), server)
    logger.info(f'Starting server. Listening on port {config["metadata_database"]["port"]}.')
    server.add_insecure_port(f'[::]:{config["metadata_database"]["port"]}')
    server.start()
    server.wait_for_termination()


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.config, "r", encoding="utf-8") as config_file:
        modyn_config = yaml.safe_load(config_file)

    logger.info("Initializing metadata database.")
    database = MetadataDatabase(modyn_config)

    logger.info("Starting metadata database.")
    run(database, modyn_config)

    logger.info("Metadata database returned, exiting.")


if __name__ == "__main__":
    main()
