import os
import pickle
from datetime import datetime
import pathlib

import torch
from benchmark_utils import create_timestamp, download_if_not_exists, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset
from tqdm import tqdm

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Huffpost")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}") 
    if args.all and args.dummyyear:
        HuffpostDownloader(args.dir).store_data_split_with_dummy_year()
    else:
        HuffpostDownloader(args.dir).store_data(args.all, args.dummyyear)


class HuffpostDownloader(Dataset):
    time_steps = [i for i in range(2012, 2019)]
    input_dim = 44
    num_classes = 11
    drive_id = "1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY"
    file_name = "huffpost.pkl"

    def __getitem__(self, idx):
        return self._dataset["title"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]

    def __len__(self):
        return len(self._dataset["category"])

    def __init__(self, data_dir: str):
        super().__init__()

        download_if_not_exists(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets
        self.path = data_dir
        self.train_path = pathlib.Path(str(self.path) + "_train")
        self.test_path = pathlib.Path(str(self.path) + "_test")
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

    def store_data(self, store_all_data: bool, add_final_dummy_year: bool) -> None:
        for year in tqdm(self._dataset):
            year_timestamp = create_timestamp(year=1970, month=1, day=year-2011)
            year_rows = []
            splits = [0, 1] if store_all_data else [0]
            for split in splits:
                for i in range(len(self._dataset[year][split]["headline"])):
                    text = self._dataset[year][split]["headline"][i]
                    label = self._dataset[year][split]["category"][i]
                    csv_row = f"{text}\t{label}"
                    year_rows.append(csv_row)

            # store the sentences
            text_file = os.path.join(self.path, f"{year}.csv")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(year_rows))

            # set timestamp
            os.utime(text_file, (year_timestamp, year_timestamp))

        if add_final_dummy_year:
            dummy_year = year + 1
            year_timestamp = create_timestamp(year=1970, month=1, day= dummy_year - 2011)
            text_file = os.path.join(self.path, f"{dummy_year}.csv")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(["dummy\t0"]))

            # set timestamp
            os.utime(text_file, (year_timestamp, year_timestamp))


        os.remove(os.path.join(self.path, "huffpost.pkl"))

    def store_data_split_with_dummy_year(self) -> None:
        for year in tqdm(self._dataset):
            year_timestamp = create_timestamp(year=1970, month=1, day=year-2011)
            year_rows = [[],[]]
            splits = [0, 1]
            for split in splits:
                for i in range(len(self._dataset[year][split]["headline"])):
                    text = self._dataset[year][split]["headline"][i]
                    label = self._dataset[year][split]["category"][i]
                    csv_row = f"{text}\t{label}"
                    year_rows[split].append(csv_row)

            # store the sentences
            train_file = os.path.join(self.train_path, f"{year}.csv")
            with open(train_file, "w", encoding="utf-8") as f:
                f.write("\n".join(year_rows[0]))
            os.utime(train_file, (year_timestamp, year_timestamp))
            
            test_file = os.path.join(self.test_path, f"{year}.csv")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("\n".join(year_rows[1]))
            os.utime(test_file, (year_timestamp, year_timestamp))

        # add final year
        dummy_year = year + 1
        year_timestamp = create_timestamp(year=1970, month=1, day= dummy_year - 2011)
        dummy_file = os.path.join(self.train_path, f"{dummy_year}.csv")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("\n".join(["dummy\t0"]))
        # set timestamp
        os.utime(dummy_file, (year_timestamp, year_timestamp))

        os.remove(os.path.join(self.path, "huffpost.pkl"))

if __name__ == "__main__":
    main()
