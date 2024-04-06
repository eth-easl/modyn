import pathlib
import os

if __name__ == '__main__':
    base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets")
    datasets = ["yearbook", "huffpost", "arxiv"]
    for d in datasets:
        dataset_dir = base_dir / d / "test"

        files = []
        for p in dataset_dir.iterdir():
            if p.is_file():
                files.append(p)
        
        for f in files:
            testdata_dir = dataset_dir / f.stem
            os.makedirs(testdata_dir, exist_ok=True)
            print(f, testdata_dir/f.name)
            os.rename(f, testdata_dir / f.name)
