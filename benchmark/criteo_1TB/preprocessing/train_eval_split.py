import logging
import logging.handlers
import concurrent.futures
import pathlib
import sys
from typing import Annotated, Optional
import random
import multiprocessing
import typer

from modyn.trainer_server.internal.dataset.extra_local_eval.binary_file_wrapper import BinaryFileWrapper

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
    # # persist logs to a file
    # file_handler = logging.FileHandler(f"split_bins_worker_{worker_id}.log", mode="a")
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(logging.Formatter("[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s"))
    # logger.addHandler(file_handler)

    for bin_file in bin_files:
        bin_file_name = bin_file.name
        original_file_wrapper = BinaryFileWrapper(
            file_path=str(bin_file),
            byteorder="little",
            record_size=160,
            label_size=4,
        )
        num_samples = original_file_wrapper.get_number_of_samples()
        train_file_path = target_train_day_dataset_path / bin_file_name
        test_file_path = target_test_day_dataset_path / bin_file_name

        all_indices = list(range(num_samples))
        random.seed(seed)
        random.shuffle(all_indices)
        test_size = max(int(num_samples * percentage / 100), 1)
        test_indices = all_indices[:test_size]
        train_indices = all_indices[test_size:]

        logger.info(f"[worker {worker_id} at day {day}]: Splitting {bin_file_name} with {num_samples} into {len(train_indices)} training samples and {len(test_indices)} test samples")
        original_file_wrapper.persist_sub_file(train_indices, str(train_file_path))
        original_file_wrapper.persist_sub_file(test_indices, str(test_file_path))


def main(
    original_criteo_path: Annotated[pathlib.Path, typer.Argument(help="The path to the original criteo dataset")],
    target_criteo_path: Annotated[pathlib.Path, typer.Argument(help="The path to save the split criteo dataset")],
    seed: Annotated[int, typer.Argument(help="The seed to use for the random number generator")],
    percentage: Annotated[int, typer.Option(help="The percentage of the dataset to use for evaluation")] = 1,
    days_up_to: Annotated[Optional[int], typer.Option(help="Only split the dataset up to this day")] = None,
    num_workers: Annotated[int, typer.Option(help="The number of workers to use for the split")] = 12,
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
