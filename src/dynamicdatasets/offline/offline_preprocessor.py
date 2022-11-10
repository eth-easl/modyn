import argparse
from datetime import datetime
import yaml
import uuid
import logging
import pathlib

from kafka import KafkaConsumer, errors
from json import loads
import pandas as pd
import grpc

from dynamicdatasets.offline.storage import Storage
from dynamicdatasets.metadata.metadata_pb2_grpc import MetadataStub
from dynamicdatasets.metadata.metadata_pb2 import Batch

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('OfflinePreprocessor')
STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor")
    parser.add_argument("config", help="Config File")
    args = parser.parse_args()
    return args


class OfflinePreprocessor:
    """
    Loading data from a dynamic source (currently a kafka stream), storing the data and
    providing the data importance class with the necessary information for its workflow
    """
    __config = None
    __data_storage = None
    __row_number = 0

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information
        """
        self.__config = config

        self.__data_storage = Storage(STORAGE_LOCATION + '/store')
        self.preprocess_function = None
        self.storable = None

    def run(self):
        """
        Run an instance of the offline preprocessor

        Currently, everything is hooked on this instance of the offline preprocessor
        and all the work happens from here.

        We constantly read from the kafka stream and if a new message arrives:

           1. read the message and log arrival
           2. run some offline processing (to be extended)
           3. write the file to storage as a json
           4. update the metadata in the database
           5. every n files that have arrived also create a new batch by shuffling
           existing batches where n is configurable
        """
        try:
            consumer = KafkaConsumer(
                self.__config['kafka']['topic'],
                bootstrap_servers=self.__config['kafka']['bootstrap_servers'],
                enable_auto_commit=True,
                # value_deserializer=lambda x: loads(x.decode('utf-8'))
                # value_deserializer=lambda x: x
            )
        except errors.NoBrokersAvailable as e:
            logger.exception(e)
            return
        for message in consumer:
            # Here's where I'm not quite sure exactly what can be passed through Kafka. 
            print('Preprocessor heard: ' + str(message.value))
            message_value = message.value

            # print('Read message from topic {0}'.format(
            #     self.__config['kafka']['topic']))

            # df = self.offline_preprocessing(message_value)

            # filename = self.__data_storage.write_dataset(
            #     uuid.uuid4(), df.to_json())

            # with grpc.insecure_channel('dynamicdatasets:50051') as channel:
            #     stub = MetadataStub(channel)
            #     stub.AddBatch(
            #         Batch(
            #             filename=filename,
            #             rows=df['row_id'].tolist()))

    def offline_preprocessing(self, message_value: str) -> pd.DataFrame:
        """
        Apply offline processing of the data to make it ready for storage

        Args:
            message_value (str): from the stream retrieved message

        Returns:
            pd.DataFrame: pandas dataframe of the processed data
        """
        df = pd.DataFrame()
        df = df.from_dict(message_value)
        rows_in_df = len(df.index)
        rows = list(range(self.__row_number, self.__row_number + rows_in_df))
        self.__row_number += rows_in_df
        df['row_id'] = rows
        return df

    def get_row_number(self):
        return self.__row_number

    def set_preprocess(self, preprocess_function):
        self.preprocess_function = preprocess_function

    def set_storable(self, storable):
        self.storable = storable 


def main():
    args = parse_args()
    config = args.config

    with open(config, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise yaml.YAMLError

    data_preprocessor = OfflinePreprocessor(parsed_yaml)
    data_preprocessor.run()


if __name__ == "__main__":
    main()
