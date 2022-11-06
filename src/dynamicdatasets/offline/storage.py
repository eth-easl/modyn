import os
import logging
import uuid

import pandas as pd
import webdataset as wds

STORAGE_LOCATION = os.getcwd()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('Storage')


class Storage:
    """
    Provide a high level abstraction for the data storage layer
    """
    __storage_location = None

    def __init__(self, storage_location):
        os.makedirs(storage_location, exist_ok=True)
        self.__storage_location = storage_location

    def write_dataset(self, batch_name: str, data: str):
        """
        Write a json batch to a tar

        Args:
            batch_name (str): unique batch name
            data (str): data as json

        Returns:
            str: corresponding stored filename
        """
        filename = f'{self.__storage_location}/{batch_name}.json'

        logger.info(f'Storing file {filename}')

        with open(filename, 'w+') as file:
            file.write(data)

        return filename

    def get_data(self, dict_data: dict[str, list[int]]) -> str:
        """
        Get the data from a tar

        Args:
            dict_data (dict[str, list[int]]): map of filename to indexes of the rows in the batch

        Returns:
            str: data as json
        """
        # TODO: implement this method as a remote procedure call
        data = self._get_data(dict_data)

        return data.to_json()

    def _get_data(self, dict_data: dict[str, list[int]]) -> pd.DataFrame:
        """
        Get the data from a tar

        Args:
            dict_data (dict[str, list[int]]): map of filename to indexes of the rows in the batch

        Returns:
            str: data as json
        """
        data = pd.DataFrame()
        for filename, indexes in dict_data.items():
            logger.info(f'Loading file {filename}')

            with open(filename, 'r') as file:
                data = file.read()

            df = pd.read_json(data)

            data = data.append(df.iloc[indexes])

        return data
