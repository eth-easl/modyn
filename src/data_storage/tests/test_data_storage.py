import unittest
import uuid
import os
import shutil
import pathlib

import numpy as np
import pandas as pd
import webdataset as wds

from datastorage import DataStorage

STORAGE_LOCATION = str(pathlib.Path(__file__).parent.parent.parent.resolve())


class TestDataStorage(unittest.TestCase):

    def setUp(self):
        pathlib.Path(STORAGE_LOCATION +
                     '/data_storage/tests/tmp').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(STORAGE_LOCATION + '/data_storage/tests/tmp')

    def test_write_dataset_to_tar(self):
        print(STORAGE_LOCATION)
        data_storage = DataStorage(STORAGE_LOCATION + '/data_storage/tmp')

        test_name = str(uuid.uuid4())
        data_storage.write_dataset_to_tar(test_name, 'TEST')

        self.assertTrue(os.path.isfile(STORAGE_LOCATION +
                        '/data_storage/tmp/' + test_name + '.tar'))

    def test_create_shuffled_batch(self):
        data_storage = DataStorage(STORAGE_LOCATION + '/data_storage/tests/tmp')

        df = pd.read_csv(STORAGE_LOCATION +
                         '/data_storage/tests/data/test.csv')

        rows_in_df = len(df.index)
        rows = list(range(0, rows_in_df))
        df['row_id'] = rows

        df1, df2, df3 = np.array_split(df, 3)

        data_storage.write_dataset_to_tar(
            'test1', df1.to_json())
        data_storage.write_dataset_to_tar(
            'test2', df2.to_json())
        data_storage.write_dataset_to_tar(
            'test3', df3.to_json())

        test_file1 = STORAGE_LOCATION + '/data_storage/tests/tmp/test1.tar'
        test_file2 = STORAGE_LOCATION + '/data_storage/tests/tmp/test2.tar'
        test_file3 = STORAGE_LOCATION + '/data_storage/tests/tmp/test3.tar'

        test_dict = {
            test_file1: [1,4,6],
            test_file2: [78,86,109],
            test_file3: [160,170,200],
        }

        result, _ = data_storage.create_shuffled_batch(test_dict)

        dataset = wds.WebDataset(result)
        for data in dataset:
            result_df = pd.read_json(data['data.json'].decode())
            self.assertCountEqual(result_df['row_id'], [1,4,6,78,86,109,160,170,200])
