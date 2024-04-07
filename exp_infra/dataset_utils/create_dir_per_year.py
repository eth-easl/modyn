import pathlib
import os
import shutil

if __name__ == '__main__':
    base_dir = pathlib.Path("/scratch/jinzhu/modyn/datasets")
    datasets = ["huffpost", "arxiv"]
    for d in datasets:
        dataset_dir = base_dir / f"{d}_test"

        files = []
        for p in dataset_dir.iterdir():
            if p.is_file():
                files.append(p)
        
        for f in files:
            mtime = os.path.getmtime(f)

            testdata_dir = dataset_dir / f.stem
            os.makedirs(testdata_dir, exist_ok=True)
            newf = testdata_dir/f.name
            print(f, newf)

            os.rename(f, newf)
            os.utime(newf, (mtime, mtime))
    
    # for d in datasets:
    #     dataset_dir = base_dir / f"{d}_test"

    #     files = []
    #     for p in dataset_dir.iterdir():
    #         if p.is_dir():
    #             files.append(p / f"{p.stem}.csv")
        
    #     for f in files:
    #         mtime = os.path.getmtime(f)

    #         newf = dataset_dir/f.name
    #         print(f, newf)

    #         shutil.copyfile(f, newf)
    #         os.utime(newf, (mtime, mtime))
