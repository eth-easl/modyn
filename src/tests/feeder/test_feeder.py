import unittest
from unittest.mock import MagicMock
import os

from mock import patch
import pandas as pd

from dynamicdatasets.feeder import Feeder


class TestDataFeeder(unittest.TestCase):

    def test_load_data(self):
        def __init__(self, config):
            self._batch_size = 10
            self._kafka_topic = 'test'
            self._interval_length = 0

        with patch.object(Feeder, '__init__', __init__):
            feeder = Feeder(None)

            feeder.write_to_kafka = MagicMock()
            test_file = '/tests/data/test.csv'

            feeder.load_data(test_file)

            pd.testing.assert_frame_equal(
                feeder.write_to_kafka.call_args[0][1],
                pd.read_csv(os.getcwd() + '/tests/data/test.csv').tail(9)
            )
