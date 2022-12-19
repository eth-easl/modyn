import logging
import yaml
import argparse
import pathlib

from modyn.backend.supervisor import Supervisor

logging.basicConfig(level=logging.NOTSET, format='[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Modyn Training Supervisor')
    parser.add_argument('pipeline', type=pathlib.Path, action="store", help="Pipeline configuration file")
    parser.add_argument('config', type=pathlib.Path, action="store", help="Modyn infrastructure configuration file")

    parser.add_argument('--start-replay-at', type=int, action="store",
                        help='This mode does not trigger on new data but just replays data starting at `TIMESTAMP` and ends all training afterwards. `TIMESTAMP` can be 0 and then just replays all data. See README for more.')

    return parser


if __name__ == '__main__':

    parser = setup_argparser()
    args = parser.parse_args()

    assert args.pipeline.is_file(), f"File does not exist: {args.pipeline}"
    assert args.config.is_file(), f"File does not exist: {args.pipeline}"

    with open(args.pipeline, "r") as f:
        pipeline_config = yaml.safe_load(f)

    with open(args.config, "r") as f:
        modyn_config = yaml.safe_load(f)

    if args.start_replay_at is not None:
        logger.info(f"Starting supervisor in experiment mode. Replay timestamp is set to {args.start_replay_at}")

    logger.info("Initializing supervisor.")
    supervisor = Supervisor(pipeline_config, modyn_config, args.start_replay_at)
    logger.info("Starting pipeline.")
    supervisor.pipeline()

    logger.info("Supervisor returned, exiting.")
