import unittest

from mock import patch
import pandas as pd

from dynamicdatasets.offline.preprocess.offline_preprocessor import OfflinePreprocessor


class TestOfflineDataPreprocessor(unittest.TestCase):

    def test_offline_preprocessing(self):
        def __init__(self, config):
            pass

        def test_preprocess(message_value):
            df = pd.DataFrame()
            df = df.from_dict(message_value)
            rows_in_df = len(df.index)
            rows = list(range(0, rows_in_df))
            df['row_id'] = rows
            return df
        with patch.object(OfflinePreprocessor, '__init__', __init__):
            offline_preprocessor = OfflinePreprocessor(None)
            offline_preprocessor.set_preprocess(lambda x: test_preprocess(x))

            self.assertEqual(offline_preprocessor.get_row_number(), 0)

            batch1 = {'name': ['Foo', 'Bar', 'Baz'],
                      'price': [42, 100, 150]
                      }

            df = pd.DataFrame(batch1)

            result_batch1_df = offline_preprocessor.offline_preprocessing(
                batch1)

            self.assertCountEqual(result_batch1_df['row_id'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
