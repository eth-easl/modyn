import argparse
from datetime import datetime
import yaml
import sys
import uuid

from kafka import KafkaConsumer
from json import loads

from dataorchestrator import DataOrchestrator
from datastorage import DataStorage


def parse_args():
    parser = argparse.ArgumentParser(description="Data Feeder")
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

            print('DataLoader: {0} Read message from topic {1}'.format(
                datetime.now(), self.config['kafka']['topic']))

            dataset = self.offline_preprocessing(message_value)

            filename = self.write_to_storage(uuid.uuid4(), dataset)

            self.update_metadata(filename)

            self.update_data_importance_server()

    def offline_preprocessing(self, message_value):
        return message_value

    def write_to_storage(self, batch_name, dataset):
        return self.data_storage.write_dataset_to_tar(batch_name, dataset)

    def update_metadata(self, filename):
        self.data_orchestrator.add_batch_to_metadata(filename)

    def update_data_importance_server(self):
        pass


def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_loader = DataLoader(parsed_yaml)
    data_loader.run()


if __name__ == "__main__":
    main()
