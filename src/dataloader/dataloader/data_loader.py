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

    def __init__(self, config: dict):
        self.config = config

        self.data_storage = DataStorage(config)
        self.data_orchestrator = DataOrchestrator(config)

    def run(self):
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

            filename = self.write_to_storage(uuid.uuid4(), df.to_json())

            self.update_metadata(filename, df.index.tolist())

    def offline_preprocessing(self, message_value):
        return pd.read_json(message_value)

    def write_to_storage(self, batch_name, dataset):
        return self.data_storage.write_dataset_to_tar(batch_name, dataset)

    def update_metadata(self, filename, rows):
        self.data_orchestrator.add_file(filename, rows)


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
