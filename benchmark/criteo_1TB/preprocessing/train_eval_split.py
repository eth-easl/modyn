import concurrent.futures
import logging
import logging.handlers
import os.path
import pathlib
import random
import sys
from typing import Annotated

import typer

RECORD_SIZE = 160
LABEL_SIZE = 4
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


def split_bins(
    target_train_day_dataset_path: pathlib.Path,
    target_test_day_dataset_path: pathlib.Path,
    bin_files: list[pathlib.Path],
    seed: int,
    percentage: int,
    worker_id: int,
    day: int,
):
    logger = logging.getLogger(__name__)

    for bin_file in bin_files:
        bin_file_name = bin_file.name
        file_size = os.path.getsize(bin_file)
        if file_size % RECORD_SIZE != 0:
            raise ValueError(f"File {bin_file} does not contain an exact number of records of size {RECORD_SIZE}")
        num_samples = int(file_size / RECORD_SIZE)
        train_file_path = target_train_day_dataset_path / bin_file_name
        test_file_path = target_test_day_dataset_path / bin_file_name

        all_indices = list(range(num_samples))
        random.seed(seed)
        random.shuffle(all_indices)
        test_size = max(int(num_samples * percentage / 100), 1)
        test_indices = all_indices[:test_size]
        train_indices = all_indices[test_size:]

        logger.info(
            f"[worker {worker_id} at day {day}]: Splitting {bin_file_name} with {num_samples} into {len(train_indices)} training samples and {len(test_indices)} test samples"
        )
        persist_sub_file(RECORD_SIZE, train_indices, bin_file, train_file_path)
        persist_sub_file(RECORD_SIZE, test_indices, bin_file, test_file_path)


def persist_sub_file(
    record_size: int,
    indices: list[int],
    source_file_path: pathlib.Path,
    target_file_path: pathlib.Path,
):
    with open(source_file_path, "rb") as source_file:
        data = source_file.read()

    with open(target_file_path, "wb") as target_file:
        for idx in indices:
            target_file.write(data[(idx * record_size) : (idx * record_size) + record_size])


def main(
    original_criteo_path: Annotated[pathlib.Path, typer.Argument(help="The path to the original criteo dataset")],
    target_criteo_path: Annotated[pathlib.Path, typer.Argument(help="The path to save the split criteo dataset")],
    seed: Annotated[int, typer.Argument(help="The seed to use for the random number generator")],
    percentage: Annotated[int, typer.Option(help="The percentage of the dataset to use for evaluation")] = 1,
    days_up_to: Annotated[int | None, typer.Option(help="Only split the dataset up to this day")] = None,
    num_workers: Annotated[int, typer.Option(help="The number of workers to use for the split")] = 32,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Original Criteo path: {original_criteo_path}")
    logger.info(f"Target Criteo path: {target_criteo_path}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Percentage: {percentage}")
    logger.info(f"Number of workers: {num_workers}")

    # if the target path does not exist, create it
    if not target_criteo_path.exists():
        logger.info(f"Creating target Criteo path: {target_criteo_path}")
        target_criteo_path.mkdir(parents=True)
    # create the train and test directories
    target_train_path = target_criteo_path / "train"
    target_test_path = target_criteo_path / "test"
    if not target_train_path.exists():
        logger.info(f"Creating target train path: {target_train_path}")
        target_train_path.mkdir(parents=True)
    if not target_test_path.exists():
        logger.info(f"Creating target test path: {target_test_path}")
        target_test_path.mkdir(parents=True)

    logger.info(f"Seed: {seed}")
    if days_up_to is None:
        logger.info("Didn't set days-up-to; Splitting the dataset up to the last day")
        days_up_to = 23

    logger.info(f"Splitting the dataset up to day {days_up_to}")
    for day in range(0, days_up_to + 1):
        logger.info(f"Splitting day {day}")
        # create the target day dataset path
        target_train_day_dataset_path = target_train_path / f"day{day}"
        if not target_train_day_dataset_path.exists():
            logger.info(f"Creating target day dataset path: {target_train_day_dataset_path}")
            target_train_day_dataset_path.mkdir(parents=True)

        target_test_day_dataset_path = target_test_path / f"day{day}"
        if not target_test_day_dataset_path.exists():
            logger.info(f"Creating target day dataset path: {target_test_day_dataset_path}")
            target_test_day_dataset_path.mkdir(parents=True)

        day_dataset_path = original_criteo_path / f"day{day}"
        logger.info(f"Day dataset path: {day_dataset_path}")
        bin_files = sorted(day_dataset_path.glob("*.bin"))
        logger.info(f"Found {len(bin_files)} binary files for day {day}")
        # distribute the binary files among the workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                bin_files_for_worker = [bin_files[i] for i in range(worker_id, len(bin_files), num_workers)]
                future = executor.submit(
                    split_bins,
                    target_train_day_dataset_path=target_train_day_dataset_path,
                    target_test_day_dataset_path=target_test_day_dataset_path,
                    bin_files=bin_files_for_worker,
                    seed=seed,
                    percentage=percentage,
                    worker_id=worker_id,
                    day=day,
                )
                futures.append(future)
            no_exceptions = True
            for future in concurrent.futures.as_completed(futures):
                if future.exception() is not None:
                    no_exceptions = False
                    logger.error(f"An error occurred: {future.exception()}")
            if not no_exceptions:
                logger.error("An error occurred while splitting the dataset")
                return
        logger.info(f"Finished splitting day {day}")


if __name__ == "__main__":
    typer.run(main)
