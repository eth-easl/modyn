import unittest

from mock import patch
import pandas as pd

from dataloader import DataLoader


class TestDataLoader(unittest.TestCase):

    def test_offline_preprocessing(self):
        def __init__(self, config):
            pass
        with patch.object(DataLoader, '__init__', __init__):
            dataloader = DataLoader(None)

            self.assertEqual(dataloader.row_number, 0)

            batch1 = {'name': ['foo', 'bar', 'baz'],
                      'price': [42, 100, 150]
                      }

            df = pd.DataFrame(batch1)

            result_batch1_df = dataloader.offline_preprocessing(df.to_json())

            self.assertEqual(dataloader.row_number, 3)
            self.assertEqual(result_batch1_df['row_id'][0], 0)
            self.assertEqual(result_batch1_df['row_id'][2], 2)

            batch2 = {'name': ['lorem', 'ipsum', 'dolor'],
                      'price': [1, 2, 3]
                      }

            df = pd.DataFrame(batch2)

            result_batch2_df = dataloader.offline_preprocessing(df.to_json())

            self.assertEqual(dataloader.row_number, 6)
            self.assertEqual(result_batch2_df['row_id'][0], 3)
            self.assertEqual(result_batch2_df['row_id'][2], 5)


if __name__ == '__main__':
    unittest.main()
