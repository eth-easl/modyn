import unittest
import uuid
import os 

from datastorage import DataStorage


class TestDataStorage(unittest.TestCase):

    def test_write_dataset_to_tar(self):
        data_storage = DataStorage()

        test_name = str(uuid.uuid4())
        data_storage.write_dataset_to_tar(test_name, 'TEST')

        self.assertTrue(os.path.isfile(os.getcwd() + '/store/' + test_name + '.tar'))
