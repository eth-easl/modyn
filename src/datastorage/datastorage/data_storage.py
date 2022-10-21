import os
from random import sample
from typing import List, Tuple
import json
import time
from pathlib import Path
import typing
import logging
import pathlib
import uuid

import pandas as pd
import webdataset as wds

STORAGE_LOCATION = os.getcwd()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())
logger = logging.getLogger('DataStorage')
handler = logging.FileHandler('DataStorage.log')
logger.addHandler(handler)


class DataStorage:
    config = None

    def __init__(self, config: dict):
        self.config = config
        os.makedirs(os.path.dirname(
            f'{STORAGE_LOCATION}/store/init.txt'), exist_ok=True)

    def write_dataset_to_tar(self, batch_name: str, data: str):
        """
        Write a json batch to a tar

        Args:
            batch_name (str): unique batch name
            data (str): data as json

        Returns:
            str: corresponding stored filename
        """
        filename = f'{STORAGE_LOCATION}/store/{batch_name}.tar'

        logging.info(f'Storing file {filename}')

        file = open(filename, 'w+')
        file.close()

        with wds.TarWriter(filename) as sink:
            sink.write({
                "__key__": "batch",
                "data.json": data
            })
        return filename

    def create_shuffled_batch(self, filenames_to_rows: typing.Dict[str, list[int]]) -> str:
        """
        Create a new shuffled batch from the specification of the input parameter

        Args:
            filenames_to_rows (typing.Dict[str, list[int]]): dictionary of the filenames and the corresponding rows that should be added to the new batch

        Returns:
            str: corresponding filename of the new batch
        """
        # TODO: Iterate over filenames, select rows and add to a new df

        new_df = None

        filename = self.write_dataset_to_tar(uuid.uuid4(), new_df.to_json())
        return filename
