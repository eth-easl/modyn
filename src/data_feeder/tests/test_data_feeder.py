import unittest
from unittest.mock import MagicMock
import os

from mock import patch
import pandas as pd

from datafeeder import DataFeeder

STORAGE_LOCATION = os.getcwd()


class TestDataFeeder(unittest.TestCase):

    def test_load_data(self):
        def __init__(self, config):
            self._batch_size = 3
            self._kafka_topic = 'test'
            self._interval_length = 0

        with patch.object(DataFeeder, '__init__', __init__):
            data_feeder = DataFeeder(None)

            data_feeder.write_to_kafka = MagicMock()
            test_file = '/data_feeder/tests/data/test.csv'

            data_feeder.load_data(test_file)

            pd.testing.assert_frame_equal(
                data_feeder.write_to_kafka.call_args[0][1],
                pd.read_csv(os.getcwd() + '/tests/data/test.csv')
            )
