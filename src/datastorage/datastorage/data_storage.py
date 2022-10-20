import os
from random import sample
from typing import List, Tuple
import json 
import time 
from pathlib import Path

import pandas as pd
import webdataset as wds
import pathlib

STORAGE_LOCATION = os.getcwd()

class DataStorage:
    config = None

    def __init__(self, config: dict):
        self.config = config
        os.makedirs(os.path.dirname(f'{STORAGE_LOCATION}/store/init.txt'), exist_ok=True)
        
    def write_dataset_to_tar(self, batch_name, data: str):
        filename = f'{STORAGE_LOCATION}/store/{batch_name}.tar'

        file = open(filename, 'w+')
        file.close()

        with wds.TarWriter(filename) as sink:
            sink.write({
                "__key__": "batch",
                "data.json": data
            })
        return filename

    # TODO: Think about how to best access the data (how to make it available for other instances to access the stored tar files)
    # This should involve only reading from the tar files directly as propagated by this instance
    # Write only datastructure, do not delete or change written files

    # TODO: Implement grpc for other instances to access this class from other nodes to store data
