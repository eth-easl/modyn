import os
import pickle

from benchmark_utils import create_timestamp, download_if_not_exists, setup_argparser_wildtime, setup_logger
from torch.utils.data import Dataset

logger = setup_logger()


def main():
    parser = setup_argparser_wildtime("Arxiv")
    args = parser.parse_args()

    logger.info(f"Downloading data to {args.dir}")
    ArXivDownloader(args.dir).store_data(args.all, args.dummyyear)


# There are some lines in the train dataset that are corrupted, i.e. the csv file wrapper cannot properly read the data.
# We remove these lines from the dataset.
corrupted_idx_dict = {
    2007: [33213],
    2008: [22489],
    2009: [64621, 165454],
    2015: [42007, 94935],
    2016: [111398],
    2019: [41309, 136814],
    2020: [102074],
    2021: [32013, 55660]
}


class ArXivDownloader(Dataset):
    time_steps = [i for i in range(2007, 2023)]
    input_dim = 55
    num_classes = 172
    drive_id = "1H5xzHHgXl8GOMonkb6ojye-Y2yIp436V"
    file_name = "arxiv.pkl"

    def __init__(self,  data_dir):
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

    def store_data(self, create_test_data: bool, add_final_dummy_year: bool):
        # create directories
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        train_dir = os.path.join(self.path, "train")
        os.makedirs(train_dir, exist_ok=True)

        if create_test_data:
            test_dir = os.path.join(self.path, "test")
            os.makedirs(test_dir, exist_ok=True)

        stats = {}

        for year in self._dataset:
            # for simplicity, instead of using years we map each day to a year from 1970
            year_timestamp = create_timestamp(year=1970, month=1, day=year-2006)

            def get_split_by_id(split: int) -> list[str]:
                rows = []
                for i in range(len(self._dataset[year][split]["title"])):
                    text = self._dataset[year][split]["title"][i].replace("\n", " ")
                    label = self._dataset[year][split]["category"][i]
                    csv_row = f"{text}\t{label}"
                    rows.append(csv_row)
                return rows

            train_year_rows = get_split_by_id(0)
            train_year_rows = self.filter_corrupted_lines(year, train_year_rows)
            train_file = os.path.join(train_dir, f"{year}.csv")
            with open(train_file, "w", encoding="utf-8") as f:
                f.write("\n".join(train_year_rows))

            # set timestamp
            os.utime(train_file, (year_timestamp, year_timestamp))

            if create_test_data:
                test_year_rows = get_split_by_id(1)
                test_file = os.path.join(test_dir, f"{year}.csv")
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(test_year_rows))

                # set timestamp
                os.utime(test_file, (year_timestamp, year_timestamp))
                stats[year] = {"train": len(train_year_rows), "test": len(test_year_rows)}
            else:
                stats[year] = {"train": len(train_year_rows)}
        with open(os.path.join(self.path, "overall_stats.json"), "w") as f:
            import json
            json.dump(stats, f, indent=4)

        if add_final_dummy_year:
            dummy_year = year + 1
            year_timestamp = create_timestamp(year=1970, month=1, day= dummy_year - 2006)
            train_dummy_file = os.path.join(train_dir, f"{dummy_year}.csv")
            with open(train_dummy_file, "w", encoding="utf-8") as f:
                f.write("\n".join(["dummy\t0"]))

            # set timestamp
            os.utime(train_dummy_file, (year_timestamp, year_timestamp))

            if create_test_data:
                test_dummy_file = os.path.join(test_dir, f"{dummy_year}.csv")
                with open(test_dummy_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(["dummy\t0"]))

                # set timestamp
                os.utime(test_dummy_file, (year_timestamp, year_timestamp))

        os.remove(os.path.join(self.path, "arxiv.pkl"))

    @staticmethod
    def filter_corrupted_lines(year, rows):
        if year in corrupted_idx_dict:
            corrupted_idx = corrupted_idx_dict[year]
            goodlines = []
            for i, l in enumerate(rows):
                if i not in corrupted_idx:
                    goodlines.append(l)
            return goodlines
        return rows


if __name__ == "__main__":
    main()
