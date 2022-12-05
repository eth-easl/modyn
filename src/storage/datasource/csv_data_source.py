import json
import uuid

import pandas as pd

from .base import BaseSource

class CSVDataSource(BaseSource):
    """
    CSVDataSource is an adapter for the CSV dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._dataloader = iter(pd.read_csv(self._config['storage']['data_source']['path'], chunksize=self._config['storage']['data_source']['batch_size']))

    def get_next(self, limit: int) -> tuple[list[str], list[str]]:
        data: list[bytes] = []
        keys: list[str] = []

        df = next(self._dataloader)

        for i in range(len(df)):
            d = {
                'data': df.iloc[i, 0:-1].to_dict(),
                'label': df.iloc[i, -1]
            }
            data_json = json.dumps(d)
            data.append(data_json)
            keys.append(uuid.uuid4().hex)

        return keys, data