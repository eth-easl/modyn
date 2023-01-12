import pathlib
import os
import json
from PIL import Image
import tensorflow as tf
import logging
import argparse
import shutil


logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description="Modyn Training Supervisor")
    parser_.add_argument("shards", type=int, action="store", help="Number of shards to create")
    parser_.add_argument("data", type=pathlib.Path, action="store", help="Path to data directory")
    parser_.add_argument("--store", action='store_true', help="Store data in shards", default=False)
    parser_.add_argument("--remove", action='store_true', help="Remove data after storing", default=False)

    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    assert args.shards > 0, f"Number of shards must be greater than 0: {args.shards}"
    assert args.store or args.remove, f"Either store or remove data"

    if args.store:
        logger.info(f"Storing data in {args.data} with {args.shards} shards")
        _store_data(args.data, args.shards)
    if args.remove:
        logger.info(f"Removing data in {args.data}")
        _remove_data(args.data)


def _store_data(data_dir: pathlib.Path, shards: int):
    # create directories
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for i in range(shards):
        os.mkdir(data_dir / f"mnist_shard_{i}")
    # download mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    samples_per_shard = len(x_train) // shards
    # store mnist dataset in png format
    for i in range(shards):
        for j in range(samples_per_shard):
            image = Image.fromarray(x_train[i*6000+j])
            image.save(data_dir / f"mnist_shard_{i}" / f"{i*6000+j}.png")
    # store labels in json format for each png an individual label field
    for i in range(shards):
        for j in range(samples_per_shard):
            with open(data_dir / f"mnist_shard_{i}" / f"{i*6000+j}.json", "w") as f:
                f.write(json.dumps({"label": int(y_train[i*6000+j])}))


def _remove_data(data_dir: pathlib.Path):
    shutil.rmtree(data_dir)

if __name__ == "__main__":
    main()
