import unittest
import uuid
import os
import shutil
import pathlib

import numpy as np
import pandas as pd
import webdataset as wds

from dynamicdatasets.offline.storage import Storage

TEMP_STORAGE_LOCATION = str(pathlib.Path(
    __file__).parent.parent.resolve()) + '/tmp'


class TestStorage(unittest.TestCase):

    def setUp(self):
        pathlib.Path(TEMP_STORAGE_LOCATION
                     ).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TEMP_STORAGE_LOCATION
                      )

    def test_write_dataset(self):
        data_storage = Storage(TEMP_STORAGE_LOCATION
                               )

        test_name = str(uuid.uuid4())
        data_storage.write_dataset(test_name, 'TEST')

        self.assertTrue(
            os.path.isfile(
                TEMP_STORAGE_LOCATION +
                '/' +
                test_name +
                '.json'))
