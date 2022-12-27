import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class GetRequest:
    def __init__(self, keys):
        self.keys = keys


class GetResponse:
    def __init__(self, value):
        self.value = value


class MockStorageServer:
    def __init__(self):
        pass

    def Get(self, request: GetRequest) -> GetResponse:
        return GetResponse(value=[])
