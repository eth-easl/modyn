import unittest

from mock import patch
import pandas as pd

from offlinedataloader import OfflineDataLoader


class TestOfflineDataLoader(unittest.TestCase):

    def test_offline_preprocessing(self):
        def __init__(self, config):
            pass
        with patch.object(OfflineDataLoader, '__init__', __init__):
            offline_dataloader = OfflineDataLoader(None)

            self.assertEqual(offline_dataloader.__row_number, 0)

            batch1 = {'name': ['foo', 'bar', 'baz'],
                      'price': [42, 100, 150]
                      }

            df = pd.DataFrame(batch1)

            result_batch1_df = offline_dataloader.offline_preprocessing(df.to_json())

            self.assertEqual(offline_dataloader.__row_number, 3)
            self.assertItemsEqual(result_batch1_df['row_id'], [0, 1, 2])

            batch2 = {'name': ['lorem', 'ipsum', 'dolor'],
                      'price': [1, 2, 3]
                      }

            df = pd.DataFrame(batch2)

            result_batch2_df = offline_dataloader.offline_preprocessing(df.to_json())

            self.assertEqual(offline_dataloader.__row_number, 6)
            self.assertItemsEqual(result_batch2_df['row_id'], [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
