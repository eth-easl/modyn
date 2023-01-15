import json
import random

# TODO: import SetRequest from metadata database & remove Mocks
from modyn.backend.metadata_processor.internal.mocks.mocks_metadata_database import (
    SetRequest,
)

# from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import (
#     SetRequest,
# )
from modyn.backend.metadata_processor.processor_strategies.abstract_processor_strategy import (
    MetadataProcessorStrategy,
)


class BasicMetadataProcessor(MetadataProcessorStrategy):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _process_post_training_metadata(
        self, training_id: int, data: str
    ) -> SetRequest:
        data_dict = json.loads(data)

        output_data = []
        output_keys = []
        output_seen = []
        output_scores = []

        for key, value in data_dict.items():
            score = random.random()
            if score > 0.5:
                output_data.append(value)
                output_keys.append(key)
                output_seen.append(True)
                output_scores.append(score)

        return {
            "keys": output_keys,
            "scores": output_scores,
            "seen": output_seen,
            "label": None,
            "data": output_data,
        }
