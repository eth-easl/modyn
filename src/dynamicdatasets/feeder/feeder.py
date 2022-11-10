import argparse
import logging
import pathlib
import time
from datetime import datetime

import pandas as pd
import yaml
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError

from experiments import builder
from dynamicdatasets.interfaces import Queryable

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())
logger = logging.getLogger('Feeder')


def parse_args():
    parser = argparse.ArgumentParser(description="Feeder")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args


class Feeder:
    """
    Class to simulate a dynamic dataset out of a static dataset from provided parameters
    """
    __config = None
    _batch_size = None
    _kafka_topic = None
    _interval_length = None

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information
        """
        self.__config = config
        self._batch_size = config['feeder']['batch_size']
        self._kafka_topic = config['kafka']['topic']
        self._interval_length = config['feeder']['interval_length']
        self.source: Queryable = None

    def write_to_kafka(self, topic_name: str, items):
        """
        Write a dataframe to a Kafka stream as a json

        Args:
            topic_name (str): kafka topic to write to
            items (pd.DataFrame): dataframe to be written to the stream
        """
        producer = KafkaProducer(
            bootstrap_servers=self.__config['kafka']['bootstrap_servers'],
            value_serializer=lambda x: x.encode('utf-8'))

        try:
            producer.send(topic_name, value=items)
        except KafkaTimeoutError as kte:
            logger.exception(
                "KafkaLogsProducer timeout sending log to Kafka: {0}".format(
                    kte))
        except KafkaError as ke:
            logger.exception(
                "KafkaLogsProducer error sending log to Kafka: {0}".format(ke))
        except Exception as e:
            logger.exception(
                "KafkaLogsProducer exception sending log to Kafka: {0}".format(e))

        print("Sent batch to Kafka")
        producer.flush()
        logger.info('Wrote messages into topic: {0}'.format(topic_name))

    def task_step(self):
        """
        Step through time to the next distribution (task). Publish those over Kafka. 
        """
        # for chunk in pd.read_csv(STORAGE_LOCATION + train_file, header=0,
        #                          chunksize=self._batch_size):
        #     self.write_to_kafka(self._kafka_topic, chunk)

        #     time.sleep(self._interval_length)

        if self.source is None:
            raise RuntimeError('You must connect a Queryable object to a feeder before you can get data!')
        data = self.source.query_next()
        self.write_to_kafka(self._kafka_topic, data)

    def connect_task_queriable(self, source):
        self.source = source


def main():
    args = parse_args()
    config = args.config

    # config_path = config/devel.yaml

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise yaml.YAMLError

    feeder = Feeder(parsed_yaml)

    feeder.load_data(parsed_yaml['feeder']['input_file'])


if __name__ == "__main__":
    main()
