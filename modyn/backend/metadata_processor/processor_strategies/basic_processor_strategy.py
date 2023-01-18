import json
import random

from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import AbstractProcessorStrategy


class BasicProcessorStrategy(MetadataProcessorStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def process_metadata(self, training_id: int, data: str) -> dict:
        data_dict = json.loads(data)

        output_keys = []
        output_data = []
        output_seen = []

        for key, value in data_dict.items():
            output_keys.append(key)
            output_data.append(value)
            output_seen.append(True)

        return {
            "keys": output_keys,
            "seen": output_seen,
            "data": output_data,
        }
