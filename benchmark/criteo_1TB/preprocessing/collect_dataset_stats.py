import json
import logging
import os.path
import pathlib
import sys
from typing import Annotated, Optional

import typer


RECORD_SIZE = 160
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def get_num_samples_in_one_day(day_path: pathlib.Path) -> int:
    bin_files = sorted(day_path.glob("*.bin"))
    # logger.info(f"day {day_path} has {len(list(bin_files))} bin files")
    num_samples = 0
    for bin_file in bin_files:
        file_size = os.path.getsize(bin_file)
        if file_size % RECORD_SIZE != 0:
            raise ValueError(f"File {bin_file} does not contain exact number of records of size {RECORD_SIZE}")
        num_samples += int(file_size / RECORD_SIZE)
    return num_samples


def get_one_dataset(dataset_path: pathlib.Path, days_up_to: int) -> list[int]:
    num_samples_by_day = []
    for day in range(days_up_to + 1):
        day_path = dataset_path / f"day{day}"
        if not day_path.exists():
            logger.info(f"Day {day} does not exist")
            continue
        num_samples = get_num_samples_in_one_day(day_path)
        logger.info(f"Day {day} has {num_samples} samples")
        num_samples_by_day.append(num_samples)
    return num_samples_by_day


def main(
    split_criteo_path: Annotated[pathlib.Path, typer.Argument(help="The path to the criteo dataset already split")],
    persist_stats: Annotated[bool, typer.Option(help="Whether to persist the stats to the dataset path")] = False,
    days_up_to: Annotated[int, typer.Option(help="Only split the dataset up to this day")] = 23,
):
    logger.info(f"Split Criteo path: {split_criteo_path}")
    logger.info(f"Persist stats: {persist_stats}")

    train_stats = get_one_dataset(split_criteo_path / "train", days_up_to)
    test_stats = get_one_dataset(split_criteo_path / "test", days_up_to)
    logger.info(f"train set size {sum(train_stats)}; test set size {sum(test_stats)}")
    if not persist_stats:
        return

    dataset_stats = {}
    for day in range(days_up_to + 1):
        dataset_stats[f"day{day}"] = {
            "train": train_stats[day],
            "test": test_stats[day],
        }

    dataset_stats["total"] = {
        "train": sum(train_stats),
        "test": sum(test_stats),
    }

    with open(split_criteo_path / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
