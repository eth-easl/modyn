import os
from random import sample
from typing import List, Tuple
import json 
import time 

import pandas as pd
import webdataset as wds
import pathlib

STORAGE_LOCATION = pathlib.Path(__file__).parent.resolve() + '/store'

class DataStorage:
    config = None

    def __init__(self, config: dict):
        self.config = config

    def add_to_metadata(self, metadata: dict):
        with open(f'{STORAGE_LOCATION}/metadata.json', 'a+') as f:
            metadata_object = json.load(f)
            metadata_object.setdefault('files', [])
            metadata_object['files'] += metadata

        with open(f'{STORAGE_LOCATION}/metadata.json', 'w') as f:
            json.dump(metadata_object, f)
        
    def write_dataset_to_tar(self, metadata: dict, csv: str):
        batch_name = metadata['batch_name']
        self.add_to_metadata(metadata)

        with wds.TarWriter(f'{STORAGE_LOCATION}/{batch_name}.tar') as sink:
            with open(csv, 'rb') as stream:
                data = stream.read()
            sink.write({
                "__key__": "batch",
                "data.csv": data
            })

    # TODO: Think about how to best access the data (how to make it available for other instances to access the stored tar files)
    # This should involve only reading from the tar files directly as propagated by this instance
    # Write only datastructure, do not delete or change written files

    # TODO: Implement grpc for other instances to access this class from other nodes to store data
