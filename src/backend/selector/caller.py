from selector_pb2_grpc import SelectorStub
from selector_pb2 import *

import grpc


channel = grpc.insecure_channel('localhost:50055')
stub = SelectorStub(channel=channel)


samples = stub.get_sample_keys(
    GetSamplesRequest(
        training_id=4,
        training_set_number=4,
        worker_id=0))
