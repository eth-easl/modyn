import logging
import os
import sys
from pathlib import Path
from typing import Any

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logging.basicConfig(format="%(asctime)s %(message)s")


class RegisterTrainingRequest:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers


class TrainingResponse:
    def __init__(self, training_id: int):
        self.training_id = training_id


class GetSamplesRequest:
    def __init__(self, pipeline_id: int, trigger_id: str, worker_id: int) -> None:
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.worker_id = worker_id


class GetSamplesResponse:
    def __init__(self, training_samples_subset: list[Any]) -> None:
        self.training_samples_subset = training_samples_subset


class MockSelectorServer:
    """Mocks the functionality of the grpc selector server."""

    def __init__(self) -> None:
        pass

    def register_pipeline(self, request: RegisterTrainingRequest) -> TrainingResponse:
        return TrainingResponse(training_id=10)

    def get_sample_keys(self, request: GetSamplesRequest) -> GetSamplesResponse:
        return GetSamplesResponse(training_samples_subset=[])
