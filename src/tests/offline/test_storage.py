import unittest
import uuid
import os
import shutil
import pathlib

import numpy as np
import pandas as pd
import webdataset as wds

from dynamicdatasets.offline.storage import Storage

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.resolve())


class TestStorage(unittest.TestCase):

    def setUp(self):
        pathlib.Path(STORAGE_LOCATION +
                     '/tmp').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(STORAGE_LOCATION + '/tmp')

    def test_write_dataset(self):
        data_storage = Storage(STORAGE_LOCATION + '/tmp')

        test_name = str(uuid.uuid4())
        data_storage.write_dataset(test_name, 'TEST')

        self.assertTrue(os.path.isfile(STORAGE_LOCATION +
                        '/tmp/' + test_name + '.json'))
