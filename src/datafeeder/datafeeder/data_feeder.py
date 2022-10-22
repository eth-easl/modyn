from datetime import datetime
import argparse
import yaml
import time
import pathlib
import os
import logging

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError, KafkaError

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())
logger = logging.getLogger('DataFeeder')
handler = logging.FileHandler('DataFeeder.log')
logger.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Data Feeder")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args


class DataFeeder:
    config = None

    def __init__(self, config: dict):
        self.config = config

    def write_to_kafka(self, topic_name: str, items: pd.DataFrame):
        """
        Write a dataframe to a Kafka stream as a json

        Args:
            topic_name (str): kafka topic to write to
            items (pd.DataFrame): dataframe to be written to the stream
        """
        producer = KafkaProducer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'], value_serializer=lambda x: x.encode('utf-8'))

        try:
            producer.send(topic_name, value=items.to_json())
        except KafkaTimeoutError as kte:
            logger.exception(
                "KafkaLogsProducer timeout sending log to Kafka: {0}".format(kte))
        except KafkaError as ke:
            logger.exception(
                "KafkaLogsProducer error sending log to Kafka: {0}".format(ke))
        except Exception as e:
            logger.exception(
                "KafkaLogsProducer exception sending log to Kafka: {0}".format(e))

        producer.flush()
        logger.info('Wrote messages into topic: {0}'.format(topic_name))

    def load_data(self, train_file: str):
        """
        Load the data from a specified training file (csv)

        Args:
            train_file (str): File to be read and sent over kafka
        """
        for chunk in pd.read_csv(STORAGE_LOCATION + train_file, header=0, chunksize=self.config['data_feeder']['batch_size']):
            self.write_to_kafka(self.config['kafka']['topic'], chunk)

            time.sleep(self.config['data_feeder']['interval_length'])


def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise yaml.YAMLError

    data_feeder = DataFeeder(parsed_yaml)

    logger.info('Starting up')

    time.sleep(1)

    logger.info('Ready to send first message')

    data_feeder.load_data(parsed_yaml['data_feeder']['input_file'])


if __name__ == "__main__":
    main()
