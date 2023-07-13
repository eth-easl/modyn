import os
import pickle

import torch
from benchmark_utils import create_timestamp, maybe_download, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset
from tqdm import tqdm

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Arxiv")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")
    ArXivDownloader(args.dir).store_data()


class ArXivDownloader(Dataset):
    time_steps = [i for i in range(2007, 2023)]
    input_dim = 55
    num_classes = 172
    drive_id = "1H5xzHHgXl8GOMonkb6ojye-Y2yIp436V"
    file_name = "arxiv.pkl"

    def __getitem__(self, idx):
        return self._dataset["title"][idx], torch.LongTensor([self._dataset["category"][idx]])[0]

    def __len__(self):
        return len(self._dataset["category"])

    def __init__(self,  data_dir):
        super().__init__()

        maybe_download(
            drive_id=self.drive_id,
            destination_dir=data_dir,
            destination_file_name=self.file_name,
        )
        datasets = pickle.load(open(os.path.join(data_dir, self.file_name), "rb"))
        assert self.time_steps == list(sorted(datasets.keys()))
        self._dataset = datasets
        self.path = data_dir

    def store_data(self):
        for year in tqdm(self._dataset):
            # for simplicity, instead of using years we map each day to a year from 1970
            year_timestamp = create_timestamp(year=1970, month=1, day=year-2006)
            year_rows = []
            for i in range(len(self._dataset[year][0]["title"])):
                text = self._dataset[year][0]["title"][i].replace("\n", " ")
                label = self._dataset[year][0]["category"][i]
                csv_row = f"{text}\t{label}"
                year_rows.append(csv_row)

            # store the year file
            text_file = os.path.join(self.path, f"{year}.csv")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("\n".join(year_rows))

            # set timestamp
            os.utime(text_file, (year_timestamp, year_timestamp))

        os.remove(os.path.join(self.path, "arxiv.pkl"))


if __name__ == "__main__":
    main()
