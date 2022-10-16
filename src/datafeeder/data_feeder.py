from datetime import datetime
import argparse
import yaml
import time

import pandas as pd
from kafka import KafkaProducer

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
        count=0
        producer = KafkaProducer(bootstrap_servers=self.config['kafka']['bootstrap_servers'])
        for message, key in items:
            producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(self.error_callback)
            count+=1
        producer.flush()
        print('{0} Wrote {1} messages into topic: {2}'.format(datetime.now(), count, topic_name))


    def load_data(self, train_file):
        for chunk in pd.read_csv(train_file, header=0, chunksize=self.config['data_feeder']['batch_size']):
            self.write_to_kafka(self.config['kafka']['topic'], chunk)

            time.sleep(self.config['data_feeder']['interval_length'])

def main():
    args = parse_args()
    config = args.experiment

    with open(config, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_feeder = DataFeeder(parsed_yaml)

    # TODO: Make input file location agnostic (should also be able to be remote) (17.10.2022)
    data_feeder.load_data(parsed_yaml['data_feeder']['input_file'])

if __name__ == "__main__":
    main()
