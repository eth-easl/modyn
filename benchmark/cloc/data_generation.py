# This code is partially adapted from https://github.com/hammoudhasan/CLDatasets/blob/main/src/downloader.py which is provided license-free
# We thank the authors for their work and hosting the data.
# This code requires the pip google-cloud-storage package

import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from shutil import copy
from typing import Annotated, Optional

import torch
import typer
from google.cloud import storage
from modyn.utils.logging import setup_logging
from modyn.utils.startup import set_start_method_spawn
from tqdm import tqdm

DAY_LENGTH_SECONDS = 24 * 60 * 60

logger = setup_logging(__name__)
set_start_method_spawn(logger)


def extract_single_zip(directory: str, target: str, zip_file: str) -> None:
    zip_path = os.path.join(directory, zip_file)
    output_dir = os.path.join(target, os.path.splitext(zip_file)[0])

    os.makedirs(output_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as e:
        logger.error(f"Error while extracing file {zip_path}")
        logger.error(e)


def main(
    dir: Annotated[Path, typer.Argument(help="Path to data directory")],
    dummy_year: Annotated[
        bool,
        typer.Option(help="Add a final dummy year to train also on the last trigger in Modyn"),
    ] = False,
    all: Annotated[
        bool,
        typer.Option(help="Store all the available data, including the validation and test sets."),
    ] = False,
    test: Annotated[bool, typer.Option(help="Enable test mode (just download one zip file)")] = False,
    skip_download: Annotated[
        bool,
        typer.Option(help="Skips the download and only (re)creates labels and timestamps."),
    ] = False,
    skip_unzip: Annotated[
        bool,
        typer.Option(help="Skips the unzipping and only (re)creates labels and timestamps."),
    ] = False,
    skip_labels: Annotated[bool, typer.Option(help="Skips the labeling")] = False,
    tmpdir: Annotated[
        Optional[Path],
        typer.Option(help="Use a different directory for storing temporary data"),
    ] = None,
    keep_zips: Annotated[bool, typer.Option(help="Keep the downloaded zipfiles")] = False,
) -> None:
    """CLOC data generation script."""

    tmpdir = tmpdir or dir
    logger.info(f"Final destination is {dir}; download destination is {tmpdir}")

    downloader = CLDatasets(dir, tmpdir, test_mode=test, keep_zips=keep_zips)
    if not skip_download:
        logger.info("Starting download")
        downloader.download_dataset()

    if not skip_unzip:
        logger.info("Starting extraction")
        downloader.extract()

    if not skip_labels:
        logger.info("Starting labeling")
        downloader.convert_labels_and_timestamps(all)

    downloader.remove_images_without_label()

    if dummy_year:
        downloader.add_dummy_year()

    logger.info("Done.")


class CLDatasets:
    """
    A class for downloading datasets from Google Cloud Storage.
    """

    def __init__(
        self,
        directory: Path,
        tmpdir: Path,
        test_mode: bool = False,
        unzip: bool = True,
        keep_zips: bool = False,
    ):
        """
        Initialize the CLDatasets object.

        Args:
            directory (str): The directory where the dataset will be saved.
        """

        self.dataset = "CLOC"
        self.directory = directory
        self.tmpdir = tmpdir
        self.unzip = unzip
        self.test_mode = test_mode
        self.keep_zips = keep_zips
        self.max_timestamp = 0
        self.example_path = ""
        self.example_label_path = ""

        self.directory.mkdir(exist_ok=True)
        self.tmpdir.mkdir(exist_ok=True)

    def extract(self) -> None:
        if self.unzip:
            self.unzip_data_files(str(self.tmpdir / "CLOC" / "data"))

    def convert_labels_and_timestamps(self, all_data: bool) -> None:
        self.convert_labels_and_timestamps_impl(
            self.tmpdir / "CLOC_torchsave_order_files" / "train_store_loc.torchSave",
            self.tmpdir / "CLOC_torchsave_order_files" / "train_labels.torchSave",
            self.tmpdir / "CLOC_torchsave_order_files" / "train_time.torchSave",
        )

        if all_data:
            logger.info("Converting all data")
            self.convert_labels_and_timestamps_impl(
                self.tmpdir / "CLOC_torchsave_order_files" / "cross_val_store_loc.torchSave",
                self.tmpdir / "CLOC_torchsave_order_files" / "cross_val_labels.torchSave",
                self.tmpdir / "CLOC_torchsave_order_files" / "cross_val_time.torchSave",
            )

    def remove_images_without_label(self) -> None:
        print("Removing images without label...")
        removed_files = 0

        image_paths = Path(self.directory).glob("**/*.jpg")
        for filename in tqdm(image_paths):
            file_path = Path(filename)
            label_path = Path(file_path.parent / f"{file_path.stem}.label")

            if not label_path.exists():
                removed_files += 1
                file_path.unlink()

        print(f"Removed {removed_files} images that do not have a label.")

    def convert_labels_and_timestamps_impl(
        self, store_loc_path: Path, labels_path: Path, timestamps_path: Path
    ) -> None:
        logger.info("Loading labels and timestamps.")
        store_loc = torch.load(store_loc_path)
        labels = torch.load(labels_path)
        timestamps = torch.load(timestamps_path)

        assert len(store_loc) == len(labels)
        assert len(store_loc) == len(timestamps)

        warned_once = False

        logger.info("Labels and timestamps loaded, applying")
        missing_files = 0
        for store_location, label, timestamp in tqdm(zip(store_loc, labels, timestamps), total=len(store_loc)):
            path = Path(self.directory + "/" + store_location.strip().replace("\n", ""))

            if not path.exists():
                if not self.test_mode:
                    raise FileExistsError(f"Cannot find file {path}")
                if not warned_once:
                    logger.warning(f"Cannot find file {path}, but we are in test mode. Will not repeat this warning.")
                    warned_once = True
                missing_files += 1
                continue

            label_path = Path(path.parent / f"{path.stem}.label")
            with open(label_path, "w+", encoding="utf-8") as file:
                file.write(str(int(label)))

            # Note: The timestamps obtained in the hd5 file are (very likely) seconds since 2004 (1072911600 GMT timestamp)
            actual_timestamp = timestamp + 1072911600
            self.max_timestamp = max(self.max_timestamp, actual_timestamp)

            self.example_label_path = label_path
            self.example_path = path

            os.utime(path, (actual_timestamp, actual_timestamp))

        logger.info(f"missing files for {store_loc_path} = {missing_files}/{len(store_loc)}")

    def add_dummy_year(self) -> None:
        dummy_path = self.directory / "dummy.jpg"
        dummy_label_path = self.directory / "dummy.label"

        assert not dummy_path.exists() and not dummy_label_path.exists()

        copy(self.example_path, dummy_path)
        copy(self.example_label_path, dummy_label_path)
        # Two years in the future
        dummy_timestamp = self.max_timestamp + (DAY_LENGTH_SECONDS * 365 * 2)

        os.utime(dummy_path, (dummy_timestamp, dummy_timestamp))

    def download_directory_from_gcloud(self, prefix: str) -> None:
        bucket_name = "cl-datasets"
        dl_dir = Path(self.tmpdir)
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name=bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
        first_zip_downloaded = False
        blobs_to_download = []

        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            if blob.name.endswith("zip"):
                if first_zip_downloaded and self.test_mode:
                    continue
                else:
                    first_zip_downloaded = True

            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            Path(dl_dir / directory).mkdir(parents=True, exist_ok=True)
            target = dl_dir / blob.name

            if not target.exists():
                blobs_to_download.append((dl_dir / blob.name, blob))
            else:
                print(f"Skipping {target} as it already exists")

        with ThreadPoolExecutor(max_workers=16) as executor, tqdm(total=len(blobs_to_download)) as pbar:
            futures_list = []

            def download_blob(target, blob):
                return blob.download_to_filename(target)

            for blob in blobs_to_download:
                future = executor.submit(download_blob, *blob)
                future.add_done_callback(lambda p: pbar.update(1))
                futures_list.append(future)

            # Wait for all tasks to complete
            for future in futures_list:
                future.result()

    def download_dataset(self) -> None:
        """
        Download the order files from Google Cloud Storage.
        """
        print("Order files are being downloaded...")
        start_time = time.time()

        self.download_directory_from_gcloud(self.dataset)

        elapsed_time = time.time() - start_time
        print("Elapsed time:", elapsed_time)

    def unzip_data_files(self, zip_file_directory: str) -> None:
        """
        Extracts the contents of zip files in a directory into nested folders.

        Args:
            directory: The path to the directory containing the zip files.

        Returns:
            None
        """

        zip_files = [file for file in os.listdir(zip_file_directory) if file.endswith(".zip")]

        with ProcessPoolExecutor(max_workers=96) as executor, tqdm(total=len(zip_files)) as pbar:
            futures_list = []
            for zip_file in zip_files:
                future = executor.submit(extract_single_zip, zip_file_directory, self.directory, zip_file)
                future.add_done_callback(lambda p: pbar.update(1))
                futures_list.append(future)

            # Wait for all tasks to complete
            for future in futures_list:
                future.result()

        if not self.keep_zips:
            # Remove zip files
            remove_command = f"rm {self.tmpdir}/{self.dataset}/data/*.zip"
            os.system(remove_command)


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
