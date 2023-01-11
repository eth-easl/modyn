import logging
import grpc
from concurrent import futures
import multiprocessing as mp
import yaml
import argparse
import pathlib

from modyn.trainer_server.grpc.generated.trainer_server_pb2_grpc import add_TrainerServerServicer_to_server
from modyn.trainer_server.grpc.trainer_server import TrainerGRPCServer


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description='Modyn Training Server')
    parser_.add_argument('config', type=pathlib.Path, action="store", help="Modyn infrastructure configuration file")

    return parser_


def main() -> None:

    mp.set_start_method('spawn')

    logging.basicConfig(level=logging.NOTSET,
                        format='[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    parser = setup_argparser()
    args = parser.parse_args()

    assert args.config.is_file(), f"File does not exist: {args.config}"

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_TrainerServerServicer_to_server(
        TrainerGRPCServer(), server)
    logger.info(f"Starting trainer server. Listening on port {config['trainer']['port']}")
    server.add_insecure_port('[::]:' + config['trainer']['port'])
    logger.info("start serving!")

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
