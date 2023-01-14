from modyn.backend.metadata_processor.metadata_processor_strategy import MetadataProcessorStrategy


class BasicMetadataProcessor(MetadataProcessorStrategy):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _process_post_training_metadata(
            self, training_id: int, data: str) -> SetRequest:
        data_dict = json.loads(data)

        output_data = []
        output_keys = []
        output_scores = []

        for key, value in data_dict.items():
            score = random.random()
            if score > 0.5:
                output_data.append(value)
                output_keys.append(key)
                output_scores.append(score)

        return {
            'data': output_data,
            'keys': output_keys,
            'scores': output_scores
        }
=======
import json
import random

from backend.odm.odm_pb2 import SetRequest
from backend.ptmp.processor.base import PostTrainingMetadataProcessor


class SimpleProcessor(PostTrainingMetadataProcessor):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _process_post_training_metadata(self, training_id: int, data: str) -> SetRequest:
        data_dict = json.loads(data)

        output_data = []
        output_keys = []
        output_scores = []

        for key, value in data_dict.items():
            score = random.random()
            if score > 0.5:
                output_data.append(value)
                output_keys.append(key)
                output_scores.append(score)

        return SetRequest(training_id=training_id, data=output_data, keys=output_keys, scores=output_scores)
>>>>>>> main:modyn/backend/ptmp/processor/simple_processor.py
