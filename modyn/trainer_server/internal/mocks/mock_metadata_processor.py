from typing import Any


class TrainingMetadataRequest:
    def __init__(
        self, pipeline_id: int, trigger_id: int, per_sample_metadata: list = [], per_trigger_metadata: Any = None
    ) -> None:
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.per_sample_metadata = per_sample_metadata
        self.per_trigger_metadata = per_trigger_metadata


class TrainingMetadataResponse:
    def __init__(self) -> None:
        pass


class PerSampleMetadata:
    def __init__(self, sample_id: str, loss: float) -> None:
        self.sample_id = sample_id
        self.loss = loss


class PerTriggerMetadata:
    def __init__(self, loss: float) -> None:
        self.loss = loss


class MockMetadataProcessorServer:
    """Mocks the functionality of the grpc metadata processor server."""

    def __init__(self) -> None:
        pass

    def send_metadata(self, request: TrainingMetadataRequest) -> TrainingMetadataResponse:
        return TrainingMetadataResponse()
