import logging
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logging.basicConfig(format="%(asctime)s %(message)s")


class TrainingMetadataRequest:
    def __init__(
        self, pipeline_id: int, trigger_id: int, per_sample_metadata: list = [], per_tigger_metadata: list = []
    ) -> None:
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.per_sample_metadata = per_sample_metadata
        self.per_tigger_metadata = per_tigger_metadata


class TrainingMetadataResponse:
    def __init__(self) -> None:
        pass


class PerSampleLoss:
    def __init__(self, sample_id: str, loss: float) -> None:
        self.sample_id = sample_id
        self.loss = loss


class MockMetadataProcessorServer:
    """Mocks the functionality of the grpc metadata processor server."""

    def __init__(self) -> None:
        pass

    def send_metadata(self, request: TrainingMetadataRequest) -> TrainingMetadataResponse:
        return TrainingMetadataResponse()
