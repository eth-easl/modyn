import pathlib
import os
import shutil

corrupted_idx_dict = {
    'huffpost': {
        "train": {
            '/scratch/jinzhu/modyn/datasets/readonly/huffpost_train/2012.csv': [1370],
            '/scratch/jinzhu/modyn/datasets/readonly/huffpost_train/2014.csv': [4923]
        },
        "test": {

        }
    },
    'arxiv': {
        "train": {
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2007.csv': [33213],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2008.csv': [22489],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2009.csv': [64621,165454],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2015.csv': [42007,94935],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2016.csv': [111398],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2019.csv': [41309,136814],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2020.csv': [102074],
            '/scratch/jinzhu/modyn/datasets/readonly/arxiv_train/2021.csv': [32013,55660]
        },
        "test": {

        }
    }
}

def remove_corrupted_lines(p: pathlib.Path, corrupted_idx: list[int], output_dir: pathlib.Path):
    with open(p, 'r') as f:
        lines = f.readlines()

    goodlines = []
    for i, l in enumerate(lines):
        if i not in corrupted_idx:
            goodlines.append(l)
    print(p, "goodlines", len(goodlines), "corrupted lines", len(corrupted_idx))
    
    with open(output_dir / p.name, 'w') as out_f:
        for l in goodlines:
            out_f.write(l)
    mtime = os.path.getmtime(p)
    os.utime(output_dir / p.name, (mtime, mtime))
    
def copy_file(p: pathlib.Path, output_dir: pathlib.Path):
    shutil.copyfile(p, output_dir / p.name)
    mtime = os.path.getmtime(p)
    os.utime(output_dir / p.name, (mtime, mtime))

if __name__ == '__main__':
    base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets/readonly")
    output_base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets")
    d = "arxiv"
  
    for suffix in ["train", "test"]:
        corrupted_d = corrupted_idx_dict[d]
        if len(corrupted_d[suffix]) > 0:
            dataset_dir = base_dir / f"{d}_{suffix}"
            output_dir = output_base_dir / f"{d}_{suffix}"
            os.makedirs(output_dir, exist_ok=True)

            files = []
            for p in dataset_dir.iterdir():
                if p.is_file():
                    if str(p) in corrupted_d[suffix]:
                        corrupted_idx = corrupted_d[suffix][str(p)]
                        remove_corrupted_lines(p, corrupted_idx, output_dir)
                    else:
                        copy_file(p, output_dir)