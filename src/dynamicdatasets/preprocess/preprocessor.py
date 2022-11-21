import uuid

import grpc

from dynamicdatasets.newqueue.newqueue_pb2_grpc import NewQueueStub
from dynamicdatasets.storage.storage_pb2_grpc import StorageStub
from dynamicdatasets.newqueue.newqueue_pb2 import AddRequest
from dynamicdatasets.storage.storage_pb2 import PutRequest


class Preprocessor:
    def __init__(self, config: dict, preprocess_function: str):
        self.__config = config
        self.__preprocess_function = preprocess_function

        newqueue_channel = grpc.insecure_channel(
            self.__config['newqueue']['hostname'] +
            ':' +
            self.__config['newqueue']['port'])
        self.__newqueue_stub = NewQueueStub(newqueue_channel)

        storage_channel = grpc.insecure_channel(
            self.__config['storage']['hostname'] +
            ':' +
            self.__config['storage']['port'])
        self.__storage_stub = StorageStub(storage_channel)

    def preprocess(self, value: list[bytes]):
        data = []
        keys = []
        for v in value:
            data.append(eval(self.__preprocess_function)(v))
            keys.append(uuid.uuid4().hex)

        self.__add_to_storage(keys, data)
        self.__add_to_newqueue(keys)

    def __add_to_newqueue(self, key: list[str]):
        self.__newqueue_stub.Add(AddRequest(keys=key))

    def __add_to_storage(self, keys: list[str], data: list[bytes]):
        self.__storage_stub.Put(PutRequest(keys=keys, value=data))
