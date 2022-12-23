import grpc
import os
import sys
from pathlib import Path
import logging

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logging.basicConfig(format='%(asctime)s %(message)s')

class RegisterTrainingRequest:

    def __init__(self, num_workers):
        self.num_workers = num_workers

class TrainingResponse:

    def __init__(self, training_id):
        self.training_id = training_id


class GetSamplesRequest:

    def __init__(self, training_id, worker_id):
        self.training_id = training_id
        self.worker_id = worker_id

class GetSamplesResponse:

    def __init__(self, training_samples_subset):
        self.training_samples_subset = training_samples_subset


class MockSelectorServer:
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self):
        pass

    def register_training(self, request: RegisterTrainingRequest) -> TrainingResponse:
        return TrainingResponse(training_id=10)

    def get_sample_keys(self, request: GetSamplesRequest) -> GetSamplesResponse:
        return GetSamplesResponse(training_samples_subset=[])
