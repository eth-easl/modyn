import argparse
from datetime import datetime
import yaml
import sys
import uuid
import logging
import pathlib

from kafka import KafkaConsumer
from json import loads
import pandas as pd

from dataorchestrator import DataOrchestrator
from datastorage import DataStorage

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('DataLoader')
handler = logging.FileHandler('DataLoader.log')
logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args


class DataLoader:
    config = None
    data_storage = None
    data_orchestrator = None
    nr_batches = 0
    row_number = 0

    def __init__(self, config: dict):
        self.config = config

        self.data_storage = DataStorage(config)
        self.data_orchestrator = DataOrchestrator(config, self.data_storage)

    def run(self):
        """
        Run an instance of the dataloader

        Currently, everything is hooked on this instance of the dataloader and all the work happens from here.

        We constantly read from the kafka stream and if a new message arrives:

           1. read the message and log arrival
           2. run some offline processing (to be extended)
           3. write the file to storage as a json
           4. update the metadata in the database
           5. every three files that have arrived also create a new batch by shuffling existing batches
        """
        consumer = KafkaConsumer(
            self.config['kafka']['topic'],
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            enable_auto_commit=True,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        for message in consumer:
            message_value = message.value

            logger.info('Read message from topic {0}'.format(
                self.config['kafka']['topic']))

            df = self.offline_preprocessing(message_value)

            filename = self.data_storage.write_dataset_to_tar(
                uuid.uuid4(), df.to_json())

            self.data_orchestrator.add_batch(filename, df['row_id'].tolist())

            self.nr_batches += 1
            if (self.nr_batches % self.config['data_orchestrator']['nr_files_update'] == 0):
                self.data_orchestrator.update_batches()

    def offline_preprocessing(self, message_value: str) -> pd.DataFrame:
        """
        Apply offline processing of the data to make it ready for storage

        Args:
            message_value (str): from the stream retrieved message

        Returns:
            pd.DataFrame: pandas dataframe of the processed data
        """
        df = pd.read_json(message_value)
        rows_in_df = len(df.index)
        rows = list(range(self.row_number, self.row_number + rows_in_df))
        self.row_number += rows_in_df
        df['row_id'] = rows
        return df


def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise yaml.YAMLError

    data_loader = DataLoader(parsed_yaml)
    data_loader.run()


if __name__ == "__main__":
    main()
