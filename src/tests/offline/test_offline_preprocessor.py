import unittest

from mock import patch
import pandas as pd

from dynamicdatasets.offline.offline_preprocessor import OfflinePreprocessor


class TestOfflineDataPreprocessor(unittest.TestCase):

    def test_offline_preprocessing(self):
        def __init__(self, config):
            pass
        with patch.object(OfflinePreprocessor, '__init__', __init__):
            offline_preprocessor = OfflinePreprocessor(None)

            self.assertEqual(offline_preprocessor.get_row_number(), 0)

            batch1 = {'name': ['foo', 'bar', 'baz'],
                      'price': [42, 100, 150]
                      }

            df = pd.DataFrame(batch1)

            result_batch1_df = offline_preprocessor.offline_preprocessing(
                batch1)

            self.assertEqual(offline_preprocessor.get_row_number(), 3)
            self.assertCountEqual(result_batch1_df['row_id'], [0, 1, 2])

            batch2 = {'name': ['lorem', 'ipsum', 'dolor'],
                      'price': [1, 2, 3]
                      }

            df = pd.DataFrame(batch2)

            result_batch2_df = offline_preprocessor.offline_preprocessing(df)

            self.assertEqual(offline_preprocessor.get_row_number(), 6)
            self.assertCountEqual(result_batch2_df['row_id'], [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
