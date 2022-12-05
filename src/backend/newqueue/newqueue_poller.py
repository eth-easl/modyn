import grpc
import time
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from storage.storage_pb2 import QueryRequest, QueryResponse
from storage.storage_pb2_grpc import StorageStub
from backend.newqueue.newqueue_pb2_grpc import NewQueueStub
from backend.newqueue.newqueue_pb2 import AddRequest


def poll(config_dict: dict):
    while True:
        time.sleep(config_dict['newqueue']['polling_interval'])
        storage_channel = grpc.insecure_channel(
            config_dict['storage']['hostname'] + ':' + config_dict['storage']['port'])
        storage_stub = StorageStub(storage_channel)

        query_result: QueryResponse = storage_stub.Query(QueryRequest())

        if len(query_result.keys) >= 0:
            newqueue_channel = grpc.insecure_channel(
                config_dict['newqueue']['hostname'] +
                ':' +
                config_dict['newqueue']['port'])
            newqueue_stub = NewQueueStub(newqueue_channel)
            newqueue_stub.Add(AddRequest(keys=query_result.keys))


if __name__ == '__main__':
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python newqueue_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    # Wait for all components to start
    time.sleep(20)
    poll(config)
