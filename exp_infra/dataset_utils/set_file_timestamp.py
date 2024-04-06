import os
import pathlib
from datetime import datetime, timezone

def create_timestamp(year: int, month: int = 1, day: int = 1) -> int:
    return int(datetime(year=year, month=month, day=day, tzinfo=timezone.utc).timestamp())

def set_one_file(f: pathlib.Path, year: int, month: int, day: int) -> int:
    t = create_timestamp(year, month, day)
    os.utime(f, (t, t))
    return t


if __name__ == '__main__':
    base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets")
    datasets = ['huffpost', 'arxiv']

    for d in datasets:
        dataset_dir = base_dir / d
        train_dir = base_dir / f"{d}_train"
        files = list(dataset_dir.iterdir())

        for i, f in enumerate(files):
            t = set_one_file(f, 1970, 1, i+1)
            print(t, os.path.getmtime(f), f)