import argparse
from datetime import datetime
import yaml
import logging
import pathlib

from kafka import KafkaConsumer, errors
import grpc

from dynamicdatasets.offline.storage.storage import Storage
from dynamicdatasets.metadata.metadata_pb2_grpc import MetadataStub
from dynamicdatasets.metadata.metadata_pb2 import AddMetadataRequest

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
    __preprocess_function = None

    def __init__(self, config: dict):
        """
        Args:
            config (dict): YAML config file with the required structure.

            See src/config/README.md for more information
        """
        self.__config = config

        self.__data_storage = Storage(STORAGE_LOCATION + '/store')
        self.__preprocess_function = None
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
        if self.__preprocess_function is None or self.storable is None:
            raise RuntimeError(
                'Must register a preprocess function and a Storable object to run!')
        try:
            consumer = KafkaConsumer(
                self.__config['kafka']['topic'],
                bootstrap_servers=self.__config['kafka']['bootstrap_servers'],
                enable_auto_commit=True,
            )
        except errors.NoBrokersAvailable as e:
            logger.exception(e)
            return
        for message in consumer:
            # Here's where I'm not quite sure exactly what can be passed
            # through Kafka.
            print('Preprocessor heard: ' + str(message.value))
            message_value = message.value

            preprocessed = self.offline_preprocessing(message_value)
            print('Preprocessed data: ' + preprocessed)

            self.__data_storage.extend(preprocessed)

            # TODO: Update metadata & Update row number:
            # with grpc.insecure_channel('dynamicdatasets:50051') as channel:
            #     stub = MetadataStub(channel)
            #     stub.AddBatch(
            #         Batch(
            #             filename=filename,
            #             rows=df['row_id'].tolist()))

    def offline_preprocessing(self, data):
        """
        Apply offline processing of the data to make it ready for storage

        Args:
            message_value (str): from the stream retrieved message

        Returns:
            pd.DataFrame: pandas dataframe of the processed data
        """
        return self.__preprocess_function(data)

    def get_row_number(self):
        return self.__row_number

    def set_preprocess(self, preprocess_function):
        self.__preprocess_function = preprocess_function

    def set_storable(self, storable):
        self.storable = storable
        self.__data_storage.set_storable(storable)

    def get_last_item(self):
        return self.__data_storage[len(self.__data_storage) - 1]
