import os
import pathlib
from datetime import datetime, timezone, timedelta

def create_timestamp(delta_day: int) -> int:
    start_date=datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    new_date = start_date + timedelta(days=delta_day)
    return int(new_date.timestamp())

def set_one_file(f: pathlib.Path, idx: int) -> int:
    t = create_timestamp(idx)
    os.utime(f, (t, t))
    return t


if __name__ == '__main__':
    base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets")
    datasets = ['huffpost', 'arxiv', 'yearbook']

    for d in datasets:
        dataset_dir = base_dir / d
        train_dir = base_dir / f"{d}_train"
        files = list(train_dir.iterdir())

        for i, f in enumerate(files):
            t = set_one_file(f, i)
            print(t, os.path.getmtime(f), f)