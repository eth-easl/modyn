# This code is partially adapted from https://github.com/hammoudhasan/CLDatasets/blob/main/src/downloader.py which is provided license-free
# We thank the authors for their work and hosting the data.

import argparse
import pathlib
from shutil import which
import os
import time
import zipfile
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

from tqdm import tqdm

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    parser_ = argparse.ArgumentParser(description=f"CLOC Benchmark Storage Script")
    parser_.add_argument(
        "dir", type=pathlib.Path, action="store", help="Path to data directory"
    )
    parser_.add_argument(
        "--dummyyear", action="store_true", help="Add a final dummy year to train also on the last trigger in Modyn"
    )
    parser_.add_argument(
        "--all", action="store_true", help="Store all the available data, including the validation and test sets."
    )
    parser_.add_argument(
        "--test", action="store_true", help="Enable test mode (just download one zip file)"
    )
    parser_.add_argument(
        "--skip_download", action="store_true", help="Skips the download and only (re)creates labels and timestamps."
    )
    return parser_


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    logger.info(f"Destination is {args.dir}")

    downloader = CLDatasets(str(args.dir), test_mode = args.test)
    if not args.skip_download:
        logger.info("Starting download and extraction.")
        downloader.download_and_extract()

    downloader.convert_labels()
    downloader.update_timestamps()
    
    if args.dummyyear:
        downloader.add_dummy_year()

    logger.info("Cleaning up")
    downloader.cleanup()


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


class CLDatasets:
    """
    A class for downloading datasets from Google Cloud Storage.
    """

    def __init__(self, directory: str, test_mode: bool = False, unzip: bool = True):
        """
        Initialize the CLDatasets object.

        Args:
            directory (str): The directory where the dataset will be saved.
        """

        self.dataset = "CLOC"
        self.directory = directory
        self.unzip = unzip
        self.test_mode = test_mode

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def download_and_extract(self):
        self.download_dataset()

        if self.unzip:
            self.unzip_data_files(self.directory + "/CLOC/data")

    def convert_labels(self):
        logger.info("Loading labels and timestamps.")
        store_loc = torch.load(self.directory + "/CLOC_torchsave_order_files/train_store_loc.torchSave")
        labels = torch.load(self.directory + "/CLOC_torchsave_order_files/train_labels.torchSave")
        timestamps = torch.load(self.directory + "/CLOC_torchsave_order_files/train_time.torchSave")

        warned_once = False

        logger.info("Labels and timestamps loaded, applying")
        for store_location, label, timestamp in tqdm(zip(store_loc, labels, timestamps), total=len(store_loc)):

            path = pathlib.Path(self.directory + "/CLOC/data/" + store_location.strip().replace("\n", ""))
            
            if not path.exists():
                if not self.test_mode:
                    raise FileExistsError(f"Cannot find file {store_location}")
                if not warned_once:
                    logger.warning(f"Cannot find file {store_location}, but we are in test mode. Will not repeat this warning.")
                    warned_once = True
                continue


            label_path = pathlib.Path(path.parent / f"{path.stem}.label")
            with open(label_path, "w+", encoding="utf-8") as file:
                file.write(str(int(label)))

            # Note: The timestamps obtained in the hd5 file are (very likely) seconds since 2004 (1072911600 GMT timestamp)
            actual_timestamp = timestamp + 1072911600
            os.utime(path, (actual_timestamp, actual_timestamp))

            


    def update_timestamps(self):
        
        pass

    def add_dummy_year(self):
        pass

    def cleanup(self):
        pass

    def download_directory_from_gcloud(self, prefix):
        bucket_name = 'cl-datasets'
        dl_dir = pathlib.Path(self.directory)
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(bucket_name=bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
        first_zip_downloaded = False
        for blob in blobs:
            print(blob.name)
            if blob.name.endswith("/"):
                continue
            if blob.name.endswith("zip"):
                if first_zip_downloaded and self.test_mode:
                    continue
                else:
                    first_zip_downloaded = True
                
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            pathlib.Path(dl_dir / directory).mkdir(parents=True, exist_ok=True)
            target = dl_dir / blob.name

            if not target.exists():
                blob.download_to_filename(dl_dir / blob.name) 
            else:
                print(f"Skipping {target} as it already exists")

    def download_dataset(self):
        """
        Download the order files from Google Cloud Storage.
        """
        print("Order files are being downloaded...")
        start_time = time.time()
        
        self.download_directory_from_gcloud(self.dataset)

        elapsed_time = time.time() - start_time
        print("Elapsed time:", elapsed_time)

    def unzip_data_files(self, directory: str) -> None:
        """
        Extracts the contents of zip files in a directory into nested folders.

        Args:
            directory: The path to the directory containing the zip files.

        Returns:
            None
        """

        zip_files = [file for file in os.listdir(directory) if file.endswith('.zip')]

        def extract_single_zip(zip_file: str) -> None:

            zip_path = os.path.join(directory, zip_file)
            output_dir = os.path.join(
                directory, os.path.splitext(zip_file)[0])

            os.makedirs(output_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

        with ThreadPoolExecutor() as executor, tqdm(total=len(zip_files)) as pbar:
            futures_list = []
            for zip_file in zip_files:
                future = executor.submit(extract_single_zip, zip_file)
                future.add_done_callback(lambda p: pbar.update(1))
                futures_list.append(future)

            # Wait for all tasks to complete
            for future in futures_list:
                future.result()

        # Remove zip files
        remove_command = f"rm {self.directory}/{self.dataset}/data/*.zip"
        os.system(remove_command)

if __name__ == "__main__":
    main()
