from datetime import datetime
import argparse
import yaml
import time
import pathlib 
import os

import pandas as pd
from kafka import KafkaProducer

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())

def parse_args():
    parser = argparse.ArgumentParser(description="Data Feeder")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args

class DataFeeder:
    config = None

    def __init__(self, config: dict):
        self.config = config

    def error_callback(self, exc):
        raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

    def write_to_kafka(self, topic_name, items):
        producer = KafkaProducer(bootstrap_servers=self.config['kafka']['bootstrap_servers'], value_serializer= lambda x: x.encode('utf-8'))
        
        producer.send(topic_name, value=items.to_json()).add_errback(self.error_callback)

        producer.flush()
        print('DataFeeder: {0} Wrote messages into topic: {1}'.format(datetime.now(), topic_name))

    def load_data(self, train_file):
        for chunk in pd.read_csv(STORAGE_LOCATION + train_file, header=0, chunksize=self.config['data_feeder']['batch_size']):
            self.write_to_kafka(self.config['kafka']['topic'], chunk)

            time.sleep(self.config['data_feeder']['interval_length'])
            break

def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_feeder = DataFeeder(parsed_yaml)

    print('DataFeeder: {0} Starting up'.format(datetime.now()))

    time.sleep(1)

    print('DataFeeder: {0} Ready to send first message'.format(datetime.now()))

    data_feeder.load_data(parsed_yaml['data_feeder']['input_file'])

if __name__ == "__main__":
    main()
