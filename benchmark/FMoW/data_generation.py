import csv
import os
import pickle
import shutil
from datetime import datetime

from benchmark.utils import maybe_download, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset
from tqdm import tqdm
from wilds import get_dataset

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("FMoW")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")

    downloader = FMOWDownloader(args.dir)
    downloader.store_data()
    downloader.clean_folder()


def parse_metadata(data_dir: str) -> list:
    filename = os.path.join(data_dir, "fmow_v1.1", "rgb_metadata.csv")
    metadata = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            picture_info = {"split": row[0], "timestamp": row[11]}
            metadata.append(picture_info)
    return metadata


class FMOWDownloader(Dataset):
    time_steps = list(range(16))
    input_dim = (3, 224, 224)
    num_classes = 62
    drive_id = "1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3"
    file_name = "fmow.pkl"

    def __init__(self, data_dir: str) -> None:
        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        self._dataset = datasets
        try:
            self._root = get_dataset(dataset="fmow", root_dir=data_dir, download=True).root
        except ValueError:
            pass
        self.metadata = parse_metadata(data_dir)
        self.data_dir = data_dir

    def clean_folder(self):
        folder_path = os.path.join(self.data_dir, "fmow_v1.1")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    def move_file_and_rename(self, index: int) -> None:
        source_dir = os.path.join(self.data_dir, "fmow_v1.1", "images")
        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            src_file = os.path.join(source_dir, f"rgb_img_{index}.png")
            dest_file = os.path.join(self.data_dir, f"rgb_img_{index}.png")
            shutil.move(src_file, dest_file)
            new_name = os.path.join(self.data_dir, f"{index}.png")
            os.rename(dest_file, new_name)

    def store_data(self):

        for year in tqdm(self._dataset):
            split = 0  # just use training split for now
            for i in range(len(self._dataset[year][split]["image_idxs"])):
                index = self._dataset[year][split]["image_idxs"][i]
                label = self._dataset[year][split]["labels"][i]
                raw_timestamp = self.metadata[index]["timestamp"]

                if len(raw_timestamp) == 24:
                    timestamp = datetime.strptime(raw_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
                else:
                    timestamp = datetime.strptime(raw_timestamp, '%Y-%m-%dT%H:%M:%SZ')

                # save label
                label_file = os.path.join(self.data_dir, f"{index}.label")
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write(str(int(label)))
                os.utime(label_file, (timestamp.timestamp(), timestamp.timestamp()))

                # set image timestamp
                self.move_file_and_rename(index)
                image_file = os.path.join(self.data_dir, f"{index}.png")
                os.utime(image_file, (timestamp.timestamp(), timestamp.timestamp()))


if __name__ == "__main__":
    main()
