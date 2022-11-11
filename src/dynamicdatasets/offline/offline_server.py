from concurrent import futures

import grpc

from dynamicdatasets.offline.offline_pb2 import DataResponse, DataRequest
from dynamicdatasets.offline.offline_pb2_grpc import OfflineServicer, add_OfflineServicer_to_server


class OfflineServicer(OfflineServicer):
    """Provides methods that implement functionality of the offline server."""

    def __init__(self, config: dict, dataset_dict: dict):
        storage_module = self.my_import('dynamicdatasets.offline.storage')
        self._storage = getattr(
            storage_module,
            config['offline']['storage'])(config)
        preprocess_module = self.my_import(
            'dynamicdatasets.offline.preprocess')
        self._preprocess = getattr(
            preprocess_module,
            config['offline']['preprocess'])(config)
        self._preprocess.set_preprocess(
            dataset_dict['preprocessor'].preprocess)
        self._preprocess.set_storable(dataset_dict['storable'])

    def my_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def GetData(self, request, context):
        print("Getting data")
        # TODO: figure out what the request object should look like
        data = self._storage.__getitem__(request)
        if data is None:
            return DataResponse(dataMap={})
        else:
            return DataResponse(dataMap=data)


def serve(config_dict, dataset_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_OfflineServicer_to_server(
        OfflineServicer(config_dict, dataset_dict), server)
    print('Starting server. Listening on port 50052.')
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()
