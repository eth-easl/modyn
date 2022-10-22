import os
import typing
import logging
import uuid
from itertools import islice

import pandas as pd
import webdataset as wds

STORAGE_LOCATION = os.getcwd()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('DataStorage')
handler = logging.FileHandler('DataStorage.log')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class DataStorage:

    def __init__(self):
        os.makedirs(f'{STORAGE_LOCATION}/store/', exist_ok=True)

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

        logger.info(f'Storing file {filename}')

        file = open(filename, 'w+')
        file.close()

        with wds.TarWriter(filename) as sink:
            sink.write({
                "__key__": "batch",
                "data.json": data
            })
        return filename

    def create_shuffled_batch(self, filenames_to_rows: dict[str, list[int]]) -> str:
        """
        Create a new shuffled batch from the specification of the input parameter

        Args:
            filenames_to_rows (dict[str, list[int]]): dictionary of the filenames and the corresponding rows that should be added to the new batch

        Returns:
            str: corresponding filename of the new batch
        """
        new_df = pd.DataFrame()
        new_rows = []
        for filename in filenames_to_rows:
            dataset = wds.WebDataset(filename)
            # TODO: The following doesn't yet properly close the tar after opening yet
            for data in islice(dataset, 0, 1):
                df = pd.read_json(data['data.json'].decode())
                selected_df = df[df['row_id'].isin(
                    filenames_to_rows[filename])]
                new_df = pd.concat([new_df, selected_df])
            new_rows.extend(filenames_to_rows[filename])
        new_df = new_df.reset_index()

        filename = self.write_dataset_to_tar(uuid.uuid4(), new_df.to_json())
        return (filename, new_df['row_id'])
