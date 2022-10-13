import os
from random import sample
from typing import List, Tuple
import json 

import webdataset as wds

# TODO: Decice on what format the metadata should be, what should be contained, how do we best store the batches, what batches
# do we cummulate together etc.

def write_dataset_to_tar(dataset: Tuple[str, dict]):
    with wds.TarWriter('out-%06d.tar') as sink:
        with open(f"{dataset[0]}", "rb") as stream:
            input = stream.read()
        sink.write({
            "__key__": "sample",
            "input.csv": input
        })
    with open("sample.json", "w") as outfile:
        json.dump(dataset[1], outfile)
    

def read_dataset_from_tar(input_tar: str):
    # TODO: Do we need to preprocess anything here?
    dataset = wds.WebDataset(input_tar)

def main():
    pass

if __name__ == "__main__":
    main()
