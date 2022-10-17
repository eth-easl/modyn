import argparse
import datetime
import yaml
from kafka import KafkaConsumer
from json import loads
from ..datastorage import DataStorage
from ..dataorchestrator import DataOrchestrator

def parse_args():
    parser = argparse.ArgumentParser(description="Data Feeder")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args

class DataLoader:
    config = None
    consumer = None
    data_storage = None
    data_orchestrator = None

    def __init__(self, config: dict):
        self.config = config
        self.consumer = KafkaConsumer(
            self.config['kafka']['topic'], 
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            auto_offstet_reset='earliest',
            value_deserializer=lambda x: loads(x.decode('utf-8'))
            )

        # TODO: check the value_deserializer (17.10.2022)
        # TODO: Test (17.10.2022)

        self.data_storage = DataStorage(config)
        self.data_orchestrator = DataOrchestrator(config)

    def run(self):
        for message in self.consumer:
            message_value = message.value

            print('DataLoader: {0} Read following message from topic {2}: {3}'.format(datetime.now(), self.config['kafka']['topic'], message_value))

            dataset = self.offline_preprocessing(message_value)

            self.write_to_storage(dataset)

            self.update_data_importance_server()

    def offline_preprocessing(self, message_value):
        return message_value

    def write_to_storage(self, dataset):
        # TODO: Replace batch name with an actual name
        self.data_storage.write_dataset_to_tar('Test', dataset)

    def update_metadata(self):
        # TODO: Replace batch name with an actual name
        self.data_orchestrator.add_batch_to_metadata('Test')

def main():
    args = parse_args()
    config = args.experiment

    with open(config, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_loader = DataLoader(parsed_yaml)
    data_loader.run()

if __name__ == "__main__":
    main()
