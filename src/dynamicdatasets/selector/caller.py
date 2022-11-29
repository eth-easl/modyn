from selector_pb2_grpc import SelectorStub
from selector_pb2 import *

import grpc


channel = grpc.insecure_channel('localhost:50055')
stub = SelectorStub(channel=channel)


# tid = stub.register_training(RegisterTrainingRequest(training_set_size=20, num_workers=1))
# print(tid)

samples = stub.get_sample_keys(
    GetSamplesRequest(
        training_id=4,
        training_set_number=4,
        worker_id=0))
print(samples.training_samples_subset)
