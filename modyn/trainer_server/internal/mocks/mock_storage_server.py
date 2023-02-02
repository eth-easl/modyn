import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class GetRequest:
    def __init__(self, dataset_id: str, keys: list[Any]) -> None:
        self.dataset_id = dataset_id
        self.keys = keys


class GetResponse:
    def __init__(self, samples: list[Any], labels: list[Any]) -> None:
        self.samples = samples
        self.labels = labels


class MockStorageServer:
    """Mocks the functionality of the grpc storage server."""

    def __init__(self) -> None:
        pass

    def Get(self, request: GetRequest) -> GetResponse:
        return GetResponse(samples=[], labels=[])
